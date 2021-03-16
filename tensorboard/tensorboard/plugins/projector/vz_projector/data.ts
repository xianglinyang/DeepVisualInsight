/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import numeric from 'numeric';
import {UMAP} from 'umap-js';

import {TSNE} from './bh_tsne';
import {SpriteMetadata} from './data-provider';
import {CameraDef} from './scatterPlot';
import * as knn from './knn';
import * as vector from './vector';
import * as logging from './logging';
import * as util from './util';

import * as searchQuery from 'search-query-parser';

export type DistanceFunction = (a: vector.Vector, b: vector.Vector) => number;
export type ProjectionComponents3D = [string, string, string];

export interface PointMetadata {
  [key: string]: number | string;
}

export interface DataProto {
  shape: [number, number];
  tensor: number[];
  metadata: {
    columns: Array<{
      name: string;
      stringValues: string[];
      numericValues: number[];
    }>;
    sprite: {
      imageBase64: string;
      singleImageDim: [number, number];
    };
  };
}

/** Statistics for a metadata column. */
export interface ColumnStats {
  name: string;
  isNumeric: boolean;
  tooManyUniqueValues: boolean;
  uniqueEntries?: Array<{
    label: string;
    count: number;
  }>;
  min: number;
  max: number;
}
export interface SpriteAndMetadataInfo {
  stats?: ColumnStats[];
  pointsInfo?: PointMetadata[];
  spriteImage?: HTMLImageElement;
  spriteMetadata?: SpriteMetadata;
}

/** A single collection of points which make up a sequence through space. */
export interface Sequence {
  /** Indices into the DataPoints array in the Data object. */
  pointIndices: number[];
}
export interface DataPoint {
  /** The point in the original space. */
  vector: Float32Array;
  /*
   * Metadata for each point. Each metadata is a set of key/value pairs
   * where the value can be a string or a number.
   */
  original_vector?: Float32Array;
  mislabel_vector?: boolean;
  color?: string;
  metadata: PointMetadata;
  /** index of the sequence, used for highlighting on click */
  sequenceIndex?: number;
  /** index in the original data source */
  index: number;
  /** This is where the calculated projections space are cached */
  projections: {
    [key: string]: number;
  };
  DVI_projections?: {
    [iteration: number]: [any, any];
  };
  DVI_color?: {
    [iteration: number]: string;
  }
  training_data?: {
    [iteration: number]: boolean;
  }
  testing_data?: {
    [iteration: number]: boolean;
  }
  current_training?: boolean;
  current_testing?: boolean;
  prediction?: {
    [iteration: number]: string;
  };
  current_prediction?: string;
  current_wrong_prediction?: boolean;
}
const IS_FIREFOX = navigator.userAgent.toLowerCase().indexOf('firefox') >= 0;
/** Controls whether nearest neighbors computation is done on the GPU or CPU. */
export const TSNE_SAMPLE_SIZE = 500;
export const UMAP_SAMPLE_SIZE = 500;
export const PCA_SAMPLE_SIZE = 50000;
/** Number of dimensions to sample when doing approximate PCA. */
export const PCA_SAMPLE_DIM = 200;
/** Number of pca components to compute. */
const NUM_PCA_COMPONENTS = 10;
/** Id of message box used for umap optimization progress bar. */
const UMAP_MSG_ID = 'umap-optimization';
/**
 * Reserved metadata attributes used for sequence information
 * NOTE: Use "__seq_next__" as "__next__" is deprecated.
 */
const SEQUENCE_METADATA_ATTRS = ['__next__', '__seq_next__'];
function getSequenceNextPointIndex(
  pointMetadata: PointMetadata
): number | null {
  let sequenceAttr = null;
  for (let metadataAttr of SEQUENCE_METADATA_ATTRS) {
    if (metadataAttr in pointMetadata && pointMetadata[metadataAttr] !== '') {
      sequenceAttr = pointMetadata[metadataAttr];
      break;
    }
  }
  if (sequenceAttr == null) {
    return null;
  }
  return +sequenceAttr;
}
/**
 * Dataset contains a DataPoints array that should be treated as immutable. This
 * acts as a working subset of the original data, with cached properties
 * from computationally expensive operations. Because creating a subset
 * requires normalizing and shifting the vector space, we make a copy of the
 * data so we can still always create new subsets based on the original data.
 */
export class DataSet {
  points: DataPoint[];
  sequences: Sequence[];
  shuffledDataIndices: number[] = [];
  /**
   * This keeps a list of all current projections so you can easily test to see
   * if it's been calculated already.
   */
  projections: {
    [projection: string]: boolean;
  } = {};
  nearest: knn.NearestEntry[][];
  spriteAndMetadataInfo: SpriteAndMetadataInfo;
  fracVariancesExplained: number[];
  tSNEIteration: number = 0;
  tSNEShouldPauseAndCheck = false;
  tSNEShouldPause = false;
  tSNEShouldStop = true;
  tSNEShouldKill = false;
  tSNEJustPause = false;
  tSNETotalIter: number = 0;
  DVIsubjectModelPath = "";
  DVIUseCache = true;
  DVIResolution = 400;
  DVIValidPointNumber: {
    [iteration: number]: number;
  } = [];
  DVICurrentRealDataNumber = 0;
  DVIRealDataNumber: {
    [iteration: number]: number;
  } = [];
  DVIEvaluation: {
    [iteration: number]: any;
  } = [];
  DVIAvailableIteration: Array<number> = [];
  DVIVisualizeDataPath = "";
  DVIFilteredData: Array<number> = [];
  superviseFactor: number;
  superviseLabels: string[];
  superviseInput: string = '';
  dim: [number, number] = [0, 0];
  hasTSNERun: boolean = false;
  private tsne: TSNE;
  hasUmapRun = false;
  private umap: UMAP;
  /** Creates a new Dataset */
  constructor(
    points: DataPoint[],
    spriteAndMetadataInfo?: SpriteAndMetadataInfo
  ) {
    this.points = points;
    this.shuffledDataIndices = util.shuffle(util.range(this.points.length));
    this.sequences = this.computeSequences(points);
    this.dim = [this.points.length, this.points[0].vector.length];
    this.spriteAndMetadataInfo = spriteAndMetadataInfo;
  }
  private computeSequences(points: DataPoint[]) {
    // Keep a list of indices seen so we don't compute sequences for a given
    // point twice.
    let indicesSeen = new Int8Array(points.length);
    // Compute sequences.
    let indexToSequence: {
      [index: number]: Sequence;
    } = {};
    let sequences: Sequence[] = [];
    for (let i = 0; i < points.length; i++) {
      if (indicesSeen[i]) {
        continue;
      }
      indicesSeen[i] = 1;
      // Ignore points without a sequence attribute.
      let next = getSequenceNextPointIndex(points[i].metadata);
      if (next == null) {
        continue;
      }
      if (next in indexToSequence) {
        let existingSequence = indexToSequence[next];
        // Pushing at the beginning of the array.
        existingSequence.pointIndices.unshift(i);
        indexToSequence[i] = existingSequence;
        continue;
      }
      // The current point is pointing to a new/unseen sequence.
      let newSequence: Sequence = {pointIndices: []};
      indexToSequence[i] = newSequence;
      sequences.push(newSequence);
      let currentIndex = i;
      while (points[currentIndex]) {
        newSequence.pointIndices.push(currentIndex);
        let next = getSequenceNextPointIndex(points[currentIndex].metadata);
        if (next != null) {
          indicesSeen[next] = 1;
          currentIndex = next;
        } else {
          currentIndex = -1;
        }
      }
    }
    return sequences;
  }
  projectionCanBeRendered(projection: ProjectionType): boolean {
    if (projection !== 'tsne') {
      return true;
    }
    return this.tSNEIteration > 0;
  }
  /**
   * Returns a new subset dataset by copying out data. We make a copy because
   * we have to modify the vectors by normalizing them.
   *
   * @param subset Array of indices of points that we want in the subset.
   *
   * @return A subset of the original dataset.
   */
  getSubset(subset?: number[]): DataSet {
    const pointsSubset =
      subset != null && subset.length > 0
        ? subset.map((i) => this.points[i])
        : this.points;
    let points = pointsSubset.map((dp) => {
      return {
        metadata: dp.metadata,
        index: dp.index,
        vector: dp.vector.slice(),
        projections: {} as {
          [key: string]: number;
        },
      };
    });
    const dp_list: DataPoint[] = [];
    for (let i = 0; i < points.length; i++) {
      const dp: DataPoint = {
        metadata: pointsSubset[i].metadata,
        index: pointsSubset[i].index,
        vector: points[i].vector,
        original_vector: pointsSubset[i].vector,
        projections: points[i].projections,
      };
      dp_list.push(dp);
    }
    return new DataSet(dp_list, this.spriteAndMetadataInfo);
  }
  /**
   * Computes the centroid, shifts all points to that centroid,
   * then makes them all unit norm.
   */
  normalize() {
    // Compute the centroid of all data points.
    let centroid = vector.centroid(this.points, (a) => a.vector);
    if (centroid == null) {
      throw Error('centroid should not be null');
    }
    // Shift all points by the centroid and make them unit norm.
    for (let id = 0; id < this.points.length; ++id) {
      let dataPoint = this.points[id];
      dataPoint.vector = vector.sub(dataPoint.vector, centroid);
      if (vector.norm2(dataPoint.vector) > 0) {
        // If we take the unit norm of a vector of all 0s, we get a vector of
        // all NaNs. We prevent that with a guard.
        vector.unit(dataPoint.vector);
      }
    }
  }
  /** Projects the dataset onto a given vector and caches the result. */
  projectLinear(dir: vector.Vector, label: string) {
    this.projections[label] = true;
    this.points.forEach((dataPoint) => {
      dataPoint.projections[label] = vector.dot(dataPoint.vector, dir);
    });
  }
  /** Projects the dataset along the top 10 principal components. */
  projectPCA(): Promise<void> {
    if (this.projections['pca-0'] != null) {
      return Promise.resolve<void>(null);
    }
    return util.runAsyncTask('Computing PCA...', () => {
      // Approximate pca vectors by sampling the dimensions.
      let dim = this.points[0].vector.length;
      let vectors = this.shuffledDataIndices.map((i) => this.points[i].vector);
      if (dim > PCA_SAMPLE_DIM) {
        vectors = vector.projectRandom(vectors, PCA_SAMPLE_DIM);
      }
      const sampledVectors = vectors.slice(0, PCA_SAMPLE_SIZE);
      const {dot, transpose, svd: numericSvd} = numeric;
      // numeric dynamically generates `numeric.div` and Closure compiler has
      // incorrectly compiles `numeric.div` property accessor. We use below
      // signature to prevent Closure from mangling and guessing.
      const div = numeric['div'];
      const scalar = dot(transpose(sampledVectors), sampledVectors);
      const sigma = div(scalar, sampledVectors.length);
      const svd = numericSvd(sigma);
      const variances: number[] = svd.S;
      let totalVariance = 0;
      for (let i = 0; i < variances.length; ++i) {
        totalVariance += variances[i];
      }
      for (let i = 0; i < variances.length; ++i) {
        variances[i] /= totalVariance;
      }
      this.fracVariancesExplained = variances;
      let U: number[][] = svd.U;
      let pcaVectors = vectors.map((vector) => {
        let newV = new Float32Array(NUM_PCA_COMPONENTS);
        for (let newDim = 0; newDim < NUM_PCA_COMPONENTS; newDim++) {
          let dot = 0;
          for (let oldDim = 0; oldDim < vector.length; oldDim++) {
            dot += vector[oldDim] * U[oldDim][newDim];
          }
          newV[newDim] = dot;
        }
        return newV;
      });
      for (let d = 0; d < NUM_PCA_COMPONENTS; d++) {
        let label = 'pca-' + d;
        this.projections[label] = true;
        for (let i = 0; i < pcaVectors.length; i++) {
          let pointIndex = this.shuffledDataIndices[i];
          this.points[pointIndex].projections[label] = pcaVectors[i][d];
        }
      }
    });
  }
  setDVIFilteredData(pointIndices: number[]) {
    this.DVIFilteredData = pointIndices;
    if(pointIndices.length > 0) {
      for (let i = 0; i < this.points.length; i++) {
        if (pointIndices.indexOf(i) == -1 && i < this.DVICurrentRealDataNumber) {
          let dataPoint = this.points[i];
          dataPoint.projections = {};
        }
      }
    } else {
      for (let i = 0; i < this.points.length; i++) {
        let dataPoint = this.points[i];
        dataPoint.projections['tsne-0'] = dataPoint.DVI_projections[this.tSNEIteration][0];
        dataPoint.projections['tsne-1'] = dataPoint.DVI_projections[this.tSNEIteration][1];
        dataPoint.projections['tsne-2'] = 0;
      }
    }
  }

  /** Runs tsne on the data. */
  async projectDVI (
      iteration: number,
      stepCallback: (iter: number, evaluation:any, totalIter?: number) => void
  ) {
    const query = 'from:hi@retrace.io,foo@gmail.com to:me subject:vacations date:1/10/2013-15/04/2014 photos:100';
    const options = {keywords: ['from', 'to', 'subject', 'Photos'], ranges: ['date']};
    const searchQueryObj = searchQuery.parse(query, options);
    console.log(searchQueryObj);
    this.projections['tsne'] = true;
    function componentToHex(c: number) {
      const hex = c.toString(16);
      return hex.length == 1 ? "0" + hex : hex;
    }

    function rgbToHex(r:number, g:number, b:number) {
      return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
    }
    if(this.DVIAvailableIteration.indexOf(iteration) == -1) {
      let headers = new Headers();
      headers.append('Content-Type', 'application/json');
      headers.append('Accept', 'application/json');
      await fetch("http://192.168.10.115:5000/animation", {
        method: 'POST',
        body: JSON.stringify({"cache": this.DVIUseCache, "path": this.DVIsubjectModelPath,  "iteration":iteration,
              "resolution":this.DVIResolution, "data_path":this.DVIVisualizeDataPath}),
        headers: headers,
        mode: 'cors'
      }).then(response => response.json()).then(data => {
        const result = data.result;
        const grid_index = data.grid_index;
        const grid_color = data.grid_color;

        const label_color_list = data.label_color_list;
        const label_list = data.label_list;
        const prediction_list = data.prediction_list;

        const background_point_number = grid_index.length;
        const real_data_number = label_color_list.length;
        this.tSNETotalIter = data.maximum_iteration;

        this.tSNEIteration = iteration;
        this.DVIValidPointNumber[iteration] = real_data_number + background_point_number;
        this.DVIAvailableIteration.push(iteration);
        const current_length = this.points.length;

        const training_data = data.training_data;
        const testing_data = data.testing_data;

        const evaluation = data.evaluation;
        this.DVIEvaluation[iteration] = evaluation;

        for (let i = 0; i < real_data_number + background_point_number - current_length; i++) {
          const newDataPoint : DataPoint = {
            metadata: {label: "background"},
            index: current_length + i,
            vector: new Float32Array(),
            projections: {
              'tsne-0': 0,
              'tsne-1': 0,
              'tsne-2': 0
            },
          };
          this.points.push(newDataPoint);
        }
        for (let i = 0; i < this.points.length; i++) {
          let dataPoint = this.points[i];
          if(dataPoint.DVI_projections == undefined || dataPoint.DVI_color == undefined) {
            dataPoint.DVI_projections = {};
            dataPoint.DVI_color = {};
          }
          if(dataPoint.training_data == undefined || dataPoint.testing_data == undefined) {
            dataPoint.training_data = {};
            dataPoint.testing_data = {};
          }
          if(dataPoint.prediction == undefined) {
            dataPoint.prediction = {};
          }
        }
        for (let i = 0; i < real_data_number; i++) {
          let dataPoint = this.points[i];
          dataPoint.projections['tsne-0'] = result[i][0];
          dataPoint.projections['tsne-1'] = result[i][1];
          dataPoint.projections['tsne-2'] = 0;
          dataPoint.color = rgbToHex(label_color_list[i][0], label_color_list[i][1], label_color_list[i][2]);
          dataPoint.DVI_projections[iteration] = [result[i][0], result[i][1]];
          dataPoint.DVI_color[iteration] = dataPoint.color;
          if (this.DVIFilteredData.length != 0 && this.DVIFilteredData.indexOf(i) == -1
              && i < this.DVICurrentRealDataNumber) {
            dataPoint.projections = {}
          }
          dataPoint.training_data[iteration] = false;
          dataPoint.testing_data[iteration] = false;
          dataPoint.current_training = false;
          dataPoint.current_testing = false;
          dataPoint.metadata['label'] = label_list[i];
          dataPoint.prediction[iteration] = prediction_list[i];
          dataPoint.current_prediction = prediction_list[i];
          if(prediction_list[i] == label_list[i]) {
            dataPoint.current_wrong_prediction = false;
          } else {
            dataPoint.current_wrong_prediction = true;
          }
        }

        for (let i = 0; i < background_point_number; i++) {
          let dataPoint = this.points[i + real_data_number];
          dataPoint.projections['tsne-0'] = grid_index[i][0];
          dataPoint.projections['tsne-1'] = grid_index[i][1];
          dataPoint.projections['tsne-2'] = 0;
          dataPoint.color = rgbToHex(grid_color[i][0],   grid_color[i][1], grid_color[i][2]);
          dataPoint.DVI_projections[iteration] = [grid_index[i][0], grid_index[i][1]];
          dataPoint.DVI_color[iteration] = dataPoint.color;
          dataPoint.training_data[iteration] = false;
          dataPoint.testing_data[iteration] = false;
          dataPoint.current_training = false;
          dataPoint.current_testing = false;
          dataPoint.prediction[iteration] = "background";
          dataPoint.current_prediction = "background";
        }

        for (let i = real_data_number + background_point_number; i < this.points.length; i++) {
          let dataPoint = this.points[i];
          dataPoint.projections = {};
        }

        for (let i = 0; i < training_data.length; i++) {
          const dataIndex = training_data[i];
          let dataPoint = this.points[dataIndex];
          dataPoint.training_data[iteration] = true;
          dataPoint.current_training = true;
        }

        for (let i = 0; i < testing_data.length; i++) {
          const dataIndex = testing_data[i];
          let dataPoint = this.points[dataIndex];
          dataPoint.testing_data[iteration] = true;
          dataPoint.current_testing = true;
        }

        this.DVICurrentRealDataNumber = real_data_number;
        this.DVIRealDataNumber[iteration] = real_data_number;
        stepCallback(this.tSNEIteration, evaluation, this.tSNETotalIter);
    });
    } else {
      const validDataNumber = this.DVIValidPointNumber[iteration];
      const evaluation = this.DVIEvaluation[iteration];
      this.tSNEIteration = iteration;
      for (let i = 0; i < validDataNumber; i++) {
        let dataPoint = this.points[i];
        dataPoint.projections['tsne-0'] = dataPoint.DVI_projections[iteration][0];
        dataPoint.projections['tsne-1'] = dataPoint.DVI_projections[iteration][1];
        dataPoint.projections['tsne-2'] = 0;
        dataPoint.color = dataPoint.DVI_color[iteration];
         if (this.DVIFilteredData.length != 0 && this.DVIFilteredData.indexOf(i) == -1
              && i < this.DVICurrentRealDataNumber) {
            dataPoint.projections = {}
          }
         dataPoint.current_training = dataPoint.training_data[iteration];
        dataPoint.current_testing = dataPoint.testing_data[iteration];
        dataPoint.current_prediction = dataPoint.current_prediction[iteration];
        if(dataPoint.current_prediction == dataPoint.metadata['label']) {
            dataPoint.current_wrong_prediction = false;
          } else {
            dataPoint.current_wrong_prediction = true;
         }
      }
      for (let i = validDataNumber; i < this.points.length; i++) {
        let dataPoint = this.points[i];
          dataPoint.projections = {};
          dataPoint.current_testing = false;
          dataPoint.current_training = false;
      }
      this.DVICurrentRealDataNumber = this.DVIRealDataNumber[iteration];
      stepCallback(this.tSNEIteration, evaluation, this.tSNETotalIter);
    }
  }

  async projectTSNE(
    perplexity: number,
    learningRate: number,
    tsneDim: number,
    stepCallback: (iter: number, dataset?:DataSet, totalIter?: number) => void
  ) {/*
    //console.log('here3');
    this.hasTSNERun = true;
    this.tSNEShouldKill = false;
    this.tSNEShouldPause = false;
    this.tSNEShouldStop = false;
    this.tSNEJustPause = false;
    this.tSNEShouldPauseAndCheck = false;
    this.tSNEIteration = 0;
    this.tSNETotalIter = 0;
    //let sampledIndices = this.shuffledDataIndices.slice(0, TSNE_SAMPLE_SIZE);
    let headers = new Headers();
    headers.append('Content-Type', 'application/json');
    headers.append('Accept', 'application/json');
    //const sampledData = sampledIndices.map((i) => this.points[i]);

    const rawdata = this.points.map((data) => {
      let datalist = [];
      for (let i = 0; i < data.original_vector.length; i++) {
        let num = data.original_vector[i];
        num = +num.toFixed(5);
        datalist.push(num)
      }
      return datalist;});
    const metadata = this.points.map((data) => data.metadata);
    let result = [[[0]]];
    let bg_list = ["0"];
    let model_prediction = [[true]];
    let grid_index = [];
    let grid_color = [];
    let label_color_list = [];
    const delay = ms => new Promise(res => setTimeout(res, ms));

    function componentToHex(c: number) {
      const hex = c.toString(16);
      return hex.length == 1 ? "0" + hex : hex;
    }

    function rgbToHex(r:number, g:number, b:number) {
      return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
    }

    let total_epoch_number = 0;
    let real_data_number = this.points.length;
    let background_point_number = 0;
    //console.log(this.points);
    let step = async () => {
      if (this.tSNEShouldKill) {
        //console.log('here2');
        return;
      }
      if (this.tSNEShouldStop || this.tSNEIteration >= total_epoch_number) {
        this.projections['tsne'] = false;
        this.tSNEJustPause = true;
        stepCallback(null);
        this.hasTSNERun = false;
        //return;
      }

      if (!(this.tSNEShouldStop || this.tSNEIteration >= total_epoch_number)
          && (!this.tSNEShouldPause || this.tSNEShouldPauseAndCheck)) {
        this.points = this.points.slice(0, real_data_number);
        //console.log(this.points);
        for (let i = 0; i < real_data_number; i++) {
          let dataPoint = this.points[i];
          dataPoint.projections['tsne-0'] = result[this.tSNEIteration][i][0];
          dataPoint.projections['tsne-1'] = result[this.tSNEIteration][i][1];
          dataPoint.projections['tsne-2'] = 0;
          dataPoint.color = rgbToHex(label_color_list[i][0], label_color_list[i][1], label_color_list[i][2])
        }
        for (let i = 0; i < background_point_number; i++) {
          const newDataPoint : DataPoint = {
            metadata: {label: "background"},
            index: real_data_number + i,
            vector: new Float32Array(),
            projections: {
              'tsne-0': grid_index[this.tSNEIteration][i][0],
              'tsne-1': grid_index[this.tSNEIteration][i][1],
              'tsne-2': 0
            },
        color: rgbToHex(grid_color[this.tSNEIteration][i][0],   grid_color[this.tSNEIteration][i][1], grid_color[this.tSNEIteration][i][2]),
        };
        this.points.push(newDataPoint);
        }
        this.projections['tsne'] = true;

        stepCallback(this.tSNEIteration + 1, undefined, new DataSet(this.points, this.spriteAndMetadataInfo),
            total_epoch_number);
        if(!this.tSNEShouldPauseAndCheck)  {
           this.tSNEIteration++;
           await delay(1000);
        }

      }
      requestAnimationFrame(step);
    };
    await fetch("http://192.168.10.115:5000/animation", {
      method: 'POST',
      body: JSON.stringify({"cache": this.DVIUseCache, "rawdata": rawdata, "metadata": metadata,
            "path": this.DVIsubjectModelPath,  "resolution":this.DVIResolution}),
      headers: headers,
      mode: 'cors'
    }).then(response => response.json()).then(data => {
      result = data.result;
      grid_index = data.grid_index;
      grid_color = data.grid_color;
      background_point_number = grid_index[0].length;
      label_color_list = data.label_color_list;
      real_data_number = label_color_list.length;
      total_epoch_number = result.length;
      this.tSNETotalIter = total_epoch_number;
      step();
    });*/
    /*
    let step = async () => {
      if (this.tSNEShouldStop || epoch >= 5) {
        this.projections['tsne'] = false;
        stepCallback(null, null);
        this.tsne = null;
        this.hasTSNERun = false;
        return;
      }
      if (!this.tSNEShouldPause) {
        sampledIndices.forEach((index, i) => {
          let dataPoint = this.points[index];
          dataPoint.projections['tsne-0'] = result[epoch][i][0];
          dataPoint.projections['tsne-1'] = result[epoch][i][1];
          if (tsneDim === 3) {
            dataPoint.projections['tsne-2'] = 0;
          }
          dataPoint.mislabel_vector = !model_prediction[epoch][i];
        });
        this.projections['tsne'] = true;
        this.tSNEIteration++;
        const bg = 'data:image/png;base64,'+ bg_list[epoch];
        epoch++;
        stepCallback(this.tSNEIteration, bg);
        await delay(10000);
      }
      requestAnimationFrame(step);
    };

    await fetch("http://192.168.10.115:5000/animation", {
      method: 'POST',
      body: JSON.stringify({"sampled_data": sampledData}),
      headers: headers,
      mode: 'cors'
    }).then(response => response.json()).then(data => {
      result = data.result;
      bg_list = data.bg_list;
      model_prediction = data.model_prediction;
      console.log(model_prediction);
      step();
    });*/
    /*
    let step = () => {
      if (this.tSNEShouldStop) {
        this.projections['tsne'] = false;
        stepCallback(null);
        this.tsne = null;
        this.hasTSNERun = false;
        return;
      }
      if (!this.tSNEShouldPause) {
        this.tsne.step();
        let result = this.tsne.getSolution();
        sampledIndices.forEach((index, i) => {
          let dataPoint = this.points[index];
          dataPoint.projections['tsne-0'] = result[i * tsneDim + 0];
          dataPoint.projections['tsne-1'] = result[i * tsneDim + 1];
          if (tsneDim === 3) {
            dataPoint.projections['tsne-2'] = result[i * tsneDim + 2];
          }
        });
        this.projections['tsne'] = true;
        this.tSNEIteration++;
        stepCallback(this.tSNEIteration);
      }
      requestAnimationFrame(step);
    };*/
    //const sampledData = sampledIndices.map((i) => this.points[i]);
    /*
    const knnComputation = this.computeKnn(sampledData, k);
    knnComputation.then((nearest) => {
      util
        .runAsyncTask('Initializing T-SNE...', () => {
          this.tsne.initDataDist(nearest);
        })
        .then(step);
    });*/
  }
  /** Runs UMAP on the data. */
  async projectUmap(
    nComponents: number,
    nNeighbors: number,
    stepCallback: (iter: number, bg:string) => void
  ) {
       this.hasUmapRun = true;
       this.umap = new UMAP({nComponents, nNeighbors});
       let currentEpoch = 0;
       const sampledIndices = this.shuffledDataIndices.slice(0, UMAP_SAMPLE_SIZE);
       const sampledData = sampledIndices.map((i) => this.points[i]);

       let headers = new Headers();
       headers.append('Content-Type', 'application/json');
       headers.append('Accept', 'application/json');

       const result_bg = await fetch("http://192.168.1.115:5000/visualize", {
         method: 'POST',
         body: JSON.stringify({"sampled_data": sampledData}),
         headers: headers,
         mode: 'cors'
       }).then(response => response.json()).then(data => [data.result, data.bg]);
       const result = result_bg[0];
       const bg = 'data:image/png;base64,'+result_bg[1];


    return new Promise((resolve, reject) => {
      util.runAsyncTask(`Updating`, () => {
        sampledIndices.forEach((index, i) => {
                  const dataPoint = this.points[index];
                  dataPoint.projections['umap-0'] = result[i][0];
                  dataPoint.projections['umap-1'] = result[i][1];
                  if (nComponents === 3) {
                    //dataPoint.projections['umap-2'] = result[i][2];
                    dataPoint.projections['umap-2'] = 0;
                  }
                });
                this.projections['umap'] = true;
                logging.setModalMessage(null, UMAP_MSG_ID);
                this.hasUmapRun = true;
                stepCallback(currentEpoch, bg);
                resolve();
      });
    });
  };
  /** Computes KNN to provide to the UMAP and t-SNE algorithms. */
  private async computeKnn(
    data: DataPoint[],
    nNeighbors: number
  ): Promise<knn.NearestEntry[][]> {
    // Handle the case where we've previously found the nearest neighbors.
    const previouslyComputedNNeighbors =
      this.nearest && this.nearest.length ? this.nearest[0].length : 0;
    if (this.nearest != null && previouslyComputedNNeighbors >= nNeighbors) {
      return Promise.resolve(
        this.nearest.map((neighbors) => neighbors.slice(0, nNeighbors))
      );
    } else {
      const knnGpuEnabled = (await util.hasWebGLSupport()) && !IS_FIREFOX;
      const result = await (knnGpuEnabled
        ? knn.findKNNGPUCosine(data, nNeighbors, (d) => d.vector)
        : knn.findKNN(
            data,
            nNeighbors,
            (d) => d.vector,
            (a, b) => vector.cosDistNorm(a, b)
          ));
      this.nearest = result;
      return Promise.resolve(result);
    }
  }
  /* Perturb TSNE and update dataset point coordinates. */
  perturbTsne() {
    if (this.hasTSNERun && this.tsne) {
      this.tsne.perturb();
      let tsneDim = this.tsne.getDim();
      let result = this.tsne.getSolution();
      let sampledIndices = this.shuffledDataIndices.slice(0, TSNE_SAMPLE_SIZE);
      sampledIndices.forEach((index, i) => {
        let dataPoint = this.points[index];
        dataPoint.projections['tsne-0'] = result[i * tsneDim + 0];
        dataPoint.projections['tsne-1'] = result[i * tsneDim + 1];
        if (tsneDim === 3) {
          dataPoint.projections['tsne-2'] = result[i * tsneDim + 2];
        }
      });
    }
  }
  setSupervision(superviseColumn: string, superviseInput?: string) {
    if (superviseColumn != null) {
      this.superviseLabels = this.shuffledDataIndices
        .slice(0, TSNE_SAMPLE_SIZE)
        .map((index) =>
          this.points[index].metadata[superviseColumn] !== undefined
            ? String(this.points[index].metadata[superviseColumn])
            : `Unknown #${index}`
        );
    }
    if (superviseInput != null) {
      this.superviseInput = superviseInput;
    }
    if (this.tsne) {
      this.tsne.setSupervision(this.superviseLabels, this.superviseInput);
    }
  }
  setSuperviseFactor(superviseFactor: number) {
    if (superviseFactor != null) {
      this.superviseFactor = superviseFactor;
      if (this.tsne) {
        this.tsne.setSuperviseFactor(superviseFactor);
      }
    }
  }
  /**
   * Merges metadata to the dataset and returns whether it succeeded.
   */
  mergeMetadata(metadata: SpriteAndMetadataInfo): boolean {
    if (metadata.pointsInfo.length !== this.points.length) {
      let errorMessage =
        `Number of tensors (${this.points.length}) do not` +
        ` match the number of lines in metadata` +
        ` (${metadata.pointsInfo.length}).`;
      if (
        metadata.stats.length === 1 &&
        this.points.length + 1 === metadata.pointsInfo.length
      ) {
        // If there is only one column of metadata and the number of points is
        // exactly one less than the number of metadata lines, this is due to an
        // unnecessary header line in the metadata and we can show a meaningful
        // error.
        logging.setErrorMessage(
          errorMessage +
            ' Single column metadata should not have a header ' +
            'row.',
          'merging metadata'
        );
        return false;
      } else if (
        metadata.stats.length > 1 &&
        this.points.length - 1 === metadata.pointsInfo.length
      ) {
        // If there are multiple columns of metadata and the number of points is
        // exactly one greater than the number of lines in the metadata, this
        // means there is a missing metadata header.
        logging.setErrorMessage(
          errorMessage +
            ' Multi-column metadata should have a header ' +
            'row with column labels.',
          'merging metadata'
        );
        return false;
      }
      logging.setWarningMessage(errorMessage);
    }
    this.spriteAndMetadataInfo = metadata;
    metadata.pointsInfo
      .slice(0, this.points.length)
      .forEach((m, i) => (this.points[i].metadata = m));
    return true;
  }
  stopTSNE() {
    this.tSNEShouldStop = true;
  }
  /**
   * Finds the nearest neighbors of the query point using a
   * user-specified distance metric.
   */
  findNeighbors(
    pointIndex: number,
    distFunc: DistanceFunction,
    numNN: number
  ): knn.NearestEntry[] {
    // Find the nearest neighbors of a particular point.
    let neighbors = knn.findKNNofPoint(
      this.points,
      pointIndex,
      numNN,
      (d) => d.vector,
      distFunc
    );
    // TODO(@dsmilkov): Figure out why we slice.
    let result = neighbors.slice(0, numNN);
    return result;
  }
  /**
   * Search the dataset based on a metadata field.
   */
  query(query: string, inRegexMode: boolean, fieldName: string): number[] {
    let predicate = util.getSearchPredicate(query, inRegexMode, fieldName);
    let matches: number[] = [];
    this.points.forEach((point, id) => {
      if (predicate(point)) {
        matches.push(id);
      }
    });
    return matches;
  }
}
export type ProjectionType = 'tsne' | 'umap' | 'pca' | 'custom';
export class Projection {
  constructor(
    public projectionType: ProjectionType,
    public projectionComponents: ProjectionComponents3D,
    public dimensionality: number,
    public dataSet: DataSet
  ) {}
}
export interface ColorOption {
  name: string;
  desc?: string;
  map?: (value: string | number) => string;
  /** List of items for the color map. Defined only for categorical map. */
  items?: {
    label: string;
    count: number;
  }[];
  /** Threshold values and their colors. Defined for gradient color map. */
  thresholds?: {
    value: number;
    color: string;
  }[];
  isSeparator?: boolean;
  tooManyUniqueValues?: boolean;
}
/**
 * An interface that holds all the data for serializing the current state of
 * the world.
 */
export class State {
  /** A label identifying this state. */
  label: string = '';
  /** Whether this State is selected in the bookmarks pane. */
  isSelected: boolean = false;
  /** The selected projection tab. */
  selectedProjection: ProjectionType;
  /** Dimensions of the DataSet. */
  dataSetDimensions: [number, number];
  /** t-SNE parameters */
  tSNEIteration: number = 0;
  tSNEPerplexity: number = 0;
  tSNELearningRate: number = 0;
  tSNEis3d: boolean = true;
  /** UMAP parameters */
  umapIs3d: boolean = true;
  umapNeighbors: number = 15;
  /** PCA projection component dimensions */
  pcaComponentDimensions: number[] = [];
  /** Custom projection parameters */
  customSelectedSearchByMetadataOption: string;
  customXLeftText: string;
  customXLeftRegex: boolean;
  customXRightText: string;
  customXRightRegex: boolean;
  customYUpText: string;
  customYUpRegex: boolean;
  customYDownText: string;
  customYDownRegex: boolean;
  /** The computed projections of the tensors. */
  projections: Array<{
    [key: string]: number;
  }> = [];
  /** Filtered dataset indices. */
  filteredPoints: number[];
  /** The indices of selected points. */
  selectedPoints: number[] = [];
  /** Camera state (2d/3d, position, target, zoom, etc). */
  cameraDef: CameraDef;
  /** Color by option. */
  selectedColorOptionName: string;
  forceCategoricalColoring: boolean;
  /** Label by option. */
  selectedLabelOption: string;
}
export function getProjectionComponents(
  projection: ProjectionType,
  components: (number | string)[]
): ProjectionComponents3D {
  if (components.length > 3) {
    throw new RangeError('components length must be <= 3');
  }
  const projectionComponents: [string, string, string] = [null, null, null];
  const prefix = projection === 'custom' ? 'linear' : projection;
  for (let i = 0; i < components.length; ++i) {
    if (components[i] == null) {
      continue;
    }
    projectionComponents[i] = prefix + '-' + components[i];
  }
  return projectionComponents;
}
export function stateGetAccessorDimensions(
  state: State
): Array<number | string> {
  let dimensions: Array<number | string>;
  switch (state.selectedProjection) {
    case 'pca':
      dimensions = state.pcaComponentDimensions.slice();
      break;
    case 'tsne':
      dimensions = [0, 1];
      if (state.tSNEis3d) {
        dimensions.push(2);
      }
      break;
    case 'umap':
      dimensions = [0, 1];
      if (state.umapIs3d) {
        dimensions.push(2);
      }
      break;
    case 'custom':
      dimensions = ['x', 'y'];
      break;
    default:
      throw new Error('Unexpected fallthrough');
  }
  return dimensions;
}
