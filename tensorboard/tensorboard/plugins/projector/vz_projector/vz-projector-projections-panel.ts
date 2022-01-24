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

import {PolymerElement} from '@polymer/polymer';
import {customElement, observe, property} from '@polymer/decorators';

import {LegacyElementMixin} from '../../../components/polymer/legacy_element_mixin';
import '../../../components/polymer/irons_and_papers';

import {template} from './vz-projector-projections-panel.html';
import './vz-projector-input';
import {
  DataSet,
  getProjectionComponents,
  PCA_SAMPLE_DIM,
  PCA_SAMPLE_SIZE,
  Projection,
  ProjectionType,
  SpriteAndMetadataInfo,
  State,
  TSNE_SAMPLE_SIZE,
  UMAP_SAMPLE_SIZE,
} from './data';
import * as vector from './vector';
import * as util from './util';
import * as logging from './logging';

const NUM_PCA_COMPONENTS = 10;

type InputControlName = 'xLeft' | 'xRight' | 'yUp' | 'yDown';
type CentroidResult = {
  centroid?: vector.Vector;
  numMatches?: number;
};
type Centroids = {
  [key: string]: vector.Vector;
  xLeft: vector.Vector;
  xRight: vector.Vector;
  yUp: vector.Vector;
  yDown: vector.Vector;
};
/**
 * A polymer component which handles the projection tabs in the projector.
 */
@customElement('vz-projector-projections-panel')
class ProjectionsPanel extends LegacyElementMixin(PolymerElement) {
  static readonly template = template;

  @property({type: Boolean})
  pcaIs3d: boolean = true;
  @property({type: Boolean})
  tSNEis3d: boolean = false;
  @property({type: Number})
  superviseFactor: number = 0;
  // UMAP parameters
  @property({type: Boolean})
  umapIs3d: boolean = true;
  @property({type: Number})
  umapNeighbors: number = 15;
  // PCA projection.
  @property({type: Array})
  pcaComponents: Array<{
    id: number;
    componentNumber: number;
    percVariance: string;
  }>;
  @property({type: Number})
  pcaX: number = 0;
  @property({type: Number})
  pcaY: number = 1;
  @property({type: Number})
  pcaZ: number = 2;
  // Custom projection.
  @property({type: String})
  customSelectedSearchByMetadataOption: string;

  @property({type: String})
  //subjectModelPathEditorInput: string = "/Users/yangxianglin/DVI_data/active_learning/random/resnet18/CIFAR10/";
  subjectModelPathEditorInput: string = "/home/xianglin/DVI_exp_data/resnet18_cifar10";
    
  @property({type: String})
  resolutionEditorInput: number;

  @property({type:Number})
  iterationEditorInput:number;

  @property({type: Boolean})
  keepSearchPredicate: boolean = true;
  // Decide wether to keep indices or search predicates, true represents search predicates

  temporalStatus: boolean = true; //true for keepSearchPredicate

  private projector: any; // Projector; type omitted b/c LegacyElement

  private currentProjection: ProjectionType;
  private polymerChangesTriggerReprojection: boolean;
  private dataSet: DataSet;
  private originalDataSet: DataSet;
  private dim: number;
  /** T-SNE perplexity. Roughly how many neighbors each point influences. */
  private perplexity: number;
  /** T-SNE learning rate. */
  private learningRate: number;
  /** T-SNE perturb interval identifier, required to terminate perturbation. */
  private perturbInterval: number;
  private searchByMetadataOptions: string[];
  /** Centroids for custom projections. */
  private centroidValues: any;
  private centroids: Centroids;
  /** The centroid across all points. */
  private allCentroid: number[];
  /** Polymer elements. */
  private runTsneButton: HTMLButtonElement;
  private pauseTsneButton: HTMLButtonElement;
  //private perturbTsneButton: HTMLButtonElement;
  private loadDVIButton: HTMLButtonElement;
  private previousDVIButton: HTMLButtonElement;
  private nextDVIButton: HTMLButtonElement;
  private jumpDVIButton: HTMLButtonElement;
  //private perplexitySlider: HTMLInputElement;
  //private learningRateInput: HTMLInputElement;
  //private superviseFactorInput: HTMLInputElement;
  private zDropdown: HTMLElement;
  private iterationLabelTsne: HTMLElement;
  private totalIterationLabelDVI: HTMLElement;
  private runUmapButton: HTMLButtonElement;
  private customProjectionXLeftInput: any; // ProjectorInput; type ommited
  private customProjectionXRightInput: any; // ProjectorInput; type ommited
  private customProjectionYUpInput: any; // ProjectorInput; type ommited
  private customProjectionYDownInput: any; // ProjectorInput; type ommited

  /*Evaluation Information*/
  private nnTrain15: HTMLElement;
  private nnTest15: HTMLElement;
  private boundTrain15: HTMLElement;
  private boundTest15: HTMLElement;
  /*
  private invNnTrain10: HTMLElement;
  private invNnTrain15: HTMLElement;
  private invNnTrain30: HTMLElement;
  private invNnTest10: HTMLElement;
  private invNnTest15: HTMLElement;
  private invNnTest30: HTMLElement;
  */
  private invAccTrain: HTMLElement;
  private invAccTest: HTMLElement;
  private invConfTrain: HTMLElement;
  private invConfTest: HTMLElement;
  private accTrain: HTMLElement;
  private accTest: HTMLElement;

  private iterationInput:number;

  initialize(projector: any) {
    this.polymerChangesTriggerReprojection = true;
    this.projector = projector;
    // Set up TSNE projections.
    this.perplexity = 30;
    this.learningRate = 10;
    // Setup Custom projections.
    this.centroidValues = {xLeft: null, xRight: null, yUp: null, yDown: null};
    this.clearCentroids();
    this.setupUIControls();
  }

  ready() {
    super.ready();
    this.zDropdown = this.$$('#z-dropdown') as HTMLElement;
    //this.runTsneButton = this.$$('.run-tsne') as HTMLButtonElement;
    //this.runTsneButton.innerText = 'Run';
   // this.pauseTsneButton = this.$$('.pause-tsne') as HTMLButtonElement;
    //this.pauseTsneButton.disabled = true;
    //this.perturbTsneButton = this.$$('.perturb-tsne') as HTMLButtonElement;
    this.previousDVIButton = this.$$('.previous-dvi') as HTMLButtonElement;
    this.previousDVIButton.disabled = true;
    this.loadDVIButton = this.$$('.load-dvi') as HTMLButtonElement;
    this.nextDVIButton = this.$$('.next-dvi') as HTMLButtonElement;
    this.nextDVIButton.disabled = true;
    this.jumpDVIButton = this.$$('.jump-dvi') as HTMLButtonElement;
    this.jumpDVIButton.disabled = true;
    //this.nextDVIButton.disabled = true;
    //this.perplexitySlider = this.$$('#perplexity-slider') as HTMLInputElement;
    /*
    this.learningRateInput = this.$$(
      '#learning-rate-slider'
    ) as HTMLInputElement;
    this.superviseFactorInput = this.$$(
      '#supervise-factor-slider'
    ) as HTMLInputElement;*/
    this.iterationLabelTsne = this.$$('.run-tsne-iter') as HTMLElement;
    this.totalIterationLabelDVI = this.$$('.dvi-total-iter') as HTMLElement;
    this.runUmapButton = this.$$('#run-umap') as HTMLButtonElement;

    /*evaluation information*/
    this.nnTrain15 = this.$$('.nn_train_15') as HTMLElement;
    this.nnTest15 = this.$$('.nn_test_15') as HTMLElement;
    this.boundTrain15 = this.$$('.bound_train_15') as HTMLElement;
    this.boundTest15 = this.$$('.bound_test_15') as HTMLElement;

    this.invAccTrain = this.$$('.inv_acc_train') as HTMLElement;
    this.invAccTest = this.$$('.inv_acc_test') as HTMLElement;
    this.invConfTrain = this.$$('.inv_conf_train') as HTMLElement;
    this.invConfTest = this.$$('.inv_conf_test') as HTMLElement;
    this.accTrain = this.$$('.acc_train') as HTMLElement;
    this.accTest = this.$$('.acc_test') as HTMLElement;
  }
  disablePolymerChangesTriggerReprojection() {
    this.polymerChangesTriggerReprojection = false;
  }
  enablePolymerChangesTriggerReprojection() {
    this.polymerChangesTriggerReprojection = true;
  }
  /*
  private updateTSNEPerplexityFromSliderChange() {
    if (this.perplexitySlider) {
      this.perplexity = +this.perplexitySlider.value;
    }
    (this.$$('.tsne-perplexity span') as HTMLSpanElement).innerText =
      '' + this.perplexity;
  }*/
  /*
  private updateTSNELearningRateFromUIChange() {
    if (this.learningRateInput) {
      this.learningRate = Math.pow(10, +this.learningRateInput.value);
    }
    (this.$$('.tsne-learning-rate span') as HTMLSpanElement).innerText =
      '' + this.learningRate;
  }
  private updateTSNESuperviseFactorFromUIChange() {
    (this.$$('.tsne-supervise-factor span') as HTMLSpanElement).innerText =
      '' + this.superviseFactor;
    if (this.dataSet) {
      this.dataSet.setSuperviseFactor(this.superviseFactor);
    }
  }*/
   private subjectModelPathEditorInputChange() {
    this.dataSet.DVIsubjectModelPath = this.subjectModelPathEditorInput;
  }
  private resolutionEditorInputChange(){
    this.dataSet.DVIResolution = this.resolutionEditorInput;
  }
  private iterationEditorInputChange(){
    this.iterationInput = Number(this.iterationEditorInput);
    console.log(this.iterationInput);
  }
  private updateEvaluationInformation(evaluation: any) {
     this.nnTrain15.innerText = ''+evaluation.nn_train_15;
     this.nnTest15.innerText = ''+evaluation.nn_test_15;
     this.boundTrain15.innerText = ''+evaluation.bound_train_15;
     this.boundTest15.innerText = ''+evaluation.bound_test_15;
     /*
     this.invNnTrain10.innerText = ''+evaluation.inv_nn_train_10;
     this.invNnTrain15.innerText = ''+evaluation.inv_nn_train_15;
     this.invNnTrain30.innerText = ''+evaluation.inv_nn_train_30;
     this.invNnTest10.innerText = ''+evaluation.inv_nn_test_10;
     this.invNnTest15.innerText = ''+evaluation.inv_nn_test_15;
     this.invNnTest30.innerText = ''+evaluation.inv_nn_test_30;
     */
     this.invAccTrain.innerText = ''+evaluation.inv_acc_train;
     this.invAccTest.innerText = ''+evaluation.inv_acc_test;
     this.invConfTrain.innerText = ''+evaluation.inv_conf_train;
     this.invConfTest.innerText = ''+evaluation.inv_conf_test;
     this.accTrain.innerText = ''+evaluation.acc_train;
     this.accTest.innerText = ''+evaluation.acc_test;
  }
  private setupUIControls() {
    {
      const self = this;
      const inkTabs = this.root.querySelectorAll('.ink-tab');
      for (let i = 0; i < inkTabs.length; i++) {
        inkTabs[i].addEventListener('click', function () {
          let id = this.getAttribute('data-tab');
          self.showTab(id);
        });
      }
    }
    /*
    this.runTsneButton.addEventListener('click', () => {
      if (this.dataSet.hasTSNERun) {
        this.dataSet.stopTSNE();
      } else {
        const delay = ms => new Promise(res => setTimeout(res, ms));

        //console.log(this.dataSet.hasTSNERun);
        this.dataSet.tSNEShouldKill = true;
        //console.log('here1');
        let act = async () => {
           await delay(500);
           this.runTSNE();
        };
        act();
      }
    });*/
    /*
    this.pauseTsneButton.addEventListener('click', () => {
      if (this.dataSet.tSNEShouldPause) {
        this.dataSet.tSNEShouldPause = false;
        this.pauseTsneButton.innerText = 'Pause';
        this.previousDVIButton.disabled = true;
        this.nextDVIButton.disabled = true;
        this.dataSet.tSNEShouldPauseAndCheck = false;
      } else {
        this.dataSet.tSNEShouldPause = true;
        this.pauseTsneButton.innerText = 'Resume';
        this.dataSet.tSNEJustPause = true;
        if (this.dataSet.tSNEIteration != 1) {
           this.previousDVIButton.disabled = false;
        }
        if (this.dataSet.tSNEIteration != this.dataSet.tSNETotalIter) {
          this.nextDVIButton.disabled = false;
        }
      }
    });*/
    this.previousDVIButton.addEventListener('click', () => {
      this.nextDVIButton.disabled = true;
      this.previousDVIButton.disabled = true;
      this.jumpDVIButton.disabled = true;
      if(this.dataSet.tSNEIteration <= 2) {
        this.previousDVIButton.disabled = true;
      }
      this.dataSet.projectDVI(this.dataSet.tSNEIteration - 1,this.projector.inspectorPanel.currentPredicate,
          (iteration: number|null, evaluation:any, new_selection:any[], indices:number[],totalIter?: number) => {
        /**
         * get filter index
         */
        //get search predicates or indices
        var filterIndices:number[];
        filterIndices = []
        if(this.temporalStatus){
          //search predicate
          this.projector.inspectorPanel.filterIndices = indices;
        }
        //indices
        filterIndices = this.projector.inspectorPanel.filterIndices;
        // TODO initilize dataset, set inspector filter indices to be all
        this.projector.dataSet.setDVIFilteredData(filterIndices);
        if (iteration != null) {
          this.iterationLabelTsne.innerText = '' + iteration;
          this.totalIterationLabelDVI.innerText = '' + totalIter;
          this.updateEvaluationInformation(evaluation);
          // this.projector.notifyProjectionPositionsUpdated(new_selection);
          this.projector.notifyProjectionPositionsUpdated();
          this.projector.onProjectionChanged();
          this.projector.onIterationChange(iteration);
        } else {
          this.projector.onProjectionChanged();
        }
        if(this.dataSet.tSNEIteration > 1) {
            this.previousDVIButton.disabled = false;
          }
        this.nextDVIButton.disabled = false;
        this.jumpDVIButton.disabled = false;
      });
    });
    this.loadDVIButton.addEventListener('click', ()=> {
      this.loadDVIButton.disabled = true;
      this.dataSet.loadprojectDVI() 
      this.nextDVIButton.disabled = true;
      this.previousDVIButton.disabled = true;
      this.jumpDVIButton.disabled = true;
      this.dataSet.projectDVI(this.dataSet.tSNEIteration + 1,this.projector.inspectorPanel.currentPredicate,
          (iteration: number|null, evaluation:any, newSelection:any[], indices:number[], totalIter?: number) => {
        /**
         * get filter index
         */
        //get search predicates or indices
        var filterIndices:number[];
        filterIndices = []
        if(this.temporalStatus){
          //search predicate
          this.projector.inspectorPanel.filterIndices = indices;
        }
        //indices
        filterIndices = this.projector.inspectorPanel.filterIndices;
        console.log(filterIndices.length);
        this.projector.dataSet.setDVIFilteredData(filterIndices);

        if (iteration != null) {
          this.iterationLabelTsne.innerText = '' + iteration;
          this.totalIterationLabelDVI.innerText = '' + totalIter;
          this.updateEvaluationInformation(evaluation);
          // this.projector.notifyProjectionPositionsUpdated(newSelection);
          this.projector.notifyProjectionPositionsUpdated();
          this.projector.onProjectionChanged();
          this.projector.onIterationChange(iteration);
          if(this.dataSet.tSNEIteration > 1) {
            this.previousDVIButton.disabled = false;
          }
          if(this.dataSet.tSNETotalIter != this.dataSet.tSNEIteration) {
            this.nextDVIButton.disabled = false;
          }
        } else {
          this.nextDVIButton.disabled = false;
          this.projector.onProjectionChanged();
        }
        this.jumpDVIButton.disabled = false;
      });
    });
    this.nextDVIButton.addEventListener('click', ()=> {
      this.nextDVIButton.disabled = true;
      this.previousDVIButton.disabled = true;
      this.jumpDVIButton.disabled = true;
      this.dataSet.projectDVI(this.dataSet.tSNEIteration + 1,this.projector.inspectorPanel.currentPredicate,
          (iteration: number|null, evaluation:any, newSelection:any[], indices:number[], totalIter?: number) => {
        /**
         * get filter index
         */
        //get search predicates or indices
        var filterIndices:number[];
        filterIndices = []
        if(this.temporalStatus){
          //search predicate
          this.projector.inspectorPanel.filterIndices = indices;
        }
        //indices
        filterIndices = this.projector.inspectorPanel.filterIndices;
        console.log(filterIndices.length);
        this.projector.dataSet.setDVIFilteredData(filterIndices);

        if (iteration != null) {
          this.iterationLabelTsne.innerText = '' + iteration;
          this.totalIterationLabelDVI.innerText = '' + totalIter;
          this.updateEvaluationInformation(evaluation);
          // this.projector.notifyProjectionPositionsUpdated(newSelection);
          this.projector.notifyProjectionPositionsUpdated();
          this.projector.onProjectionChanged();
          this.projector.onIterationChange(iteration);
          if(this.dataSet.tSNEIteration > 1) {
            this.previousDVIButton.disabled = false;
          }
          if(this.dataSet.tSNETotalIter != this.dataSet.tSNEIteration) {
            this.nextDVIButton.disabled = false;
          }
        } else {
          this.nextDVIButton.disabled = false;
          this.projector.onProjectionChanged();
        }
        this.jumpDVIButton.disabled = false;
      });
    });
    this.jumpDVIButton.addEventListener('click', ()=> {
      this.jumpDVIButton.disabled = true;
      if(this.iterationInput > this.dataSet.tSNETotalIter||this.iterationInput<1){
        logging.setErrorMessage("Invaild Input!", null);
        this.jumpDVIButton.disabled = false;
        return;
      }else if(this.iterationInput == this.dataSet.tSNEIteration){
        logging.setWarningMessage("current iteration!");
        this.jumpDVIButton.disabled = false;
        return;
      }
      this.nextDVIButton.disabled = true;
      this.previousDVIButton.disabled = true;
      this.dataSet.projectDVI(this.iterationInput,this.projector.inspectorPanel.currentPredicate,
          (iteration: number|null, evaluation:any, newSelection:any[], indices:number[], totalIter?: number) => {
        /**
         * get filter index
         */
        //get search predicates or indices
        var filterIndices:number[];
        filterIndices = []
        if(this.temporalStatus){
          //search predicate
          this.projector.inspectorPanel.filterIndices = indices;
        }
        //indices
        filterIndices = this.projector.inspectorPanel.filterIndices;
        console.log(filterIndices.length);
        this.projector.dataSet.setDVIFilteredData(filterIndices);

        if (iteration != null) {
          this.iterationLabelTsne.innerText = '' + iteration;
          this.totalIterationLabelDVI.innerText = '' + totalIter;
          this.updateEvaluationInformation(evaluation);
          // this.projector.notifyProjectionPositionsUpdated(newSelection);
          this.projector.notifyProjectionPositionsUpdated();
          this.projector.onProjectionChanged();
          this.projector.onIterationChange(iteration);
          if(this.dataSet.tSNEIteration > 1) {
            this.previousDVIButton.disabled = false;
          }
          if(this.dataSet.tSNETotalIter != this.dataSet.tSNEIteration) {
            this.nextDVIButton.disabled = false;
          }
        } else {
          this.nextDVIButton.disabled = false;
          this.projector.onProjectionChanged();
        }
        this.jumpDVIButton.disabled = false;
      });
    });

    /*
    this.nextDVIButton.addEventListener('click', () => {
      if (this.dataSet.tSNEJustPause) {
        this.dataSet.tSNEJustPause = false;
      } else {
        this.dataSet.tSNEIteration ++;
      }
      this.dataSet.tSNEShouldPauseAndCheck = true;
      if(this.dataSet.tSNEIteration == this.dataSet.tSNETotalIter) {
        this.nextDVIButton.disabled = true;
      }
      if(!this.dataSet.hasTSNERun) {
        this.runTsneButton.innerText = 'Stop';
        this.runTsneButton.disabled = false;
        this.pauseTsneButton.innerText = 'Resume';
        this.pauseTsneButton.disabled = false;
        this.dataSet.tSNEShouldStop = false;
        this.dataSet.tSNEShouldPause = true;
        this.dataSet.hasTSNERun = true;
      }
      this.previousDVIButton.disabled = false;
    });*/
    /*
    this.perturbTsneButton.addEventListener('mousedown', () => {
      if (this.dataSet && this.projector) {
        this.dataSet.perturbTsne();
        this.projector.notifyProjectionPositionsUpdated();
        this.perturbInterval = window.setInterval(() => {
          this.dataSet.perturbTsne();
          this.projector.notifyProjectionPositionsUpdated();
        }, 100);
      }
    });
    this.perturbTsneButton.addEventListener('mouseup', () => {
      clearInterval(this.perturbInterval);
    });*/
    /*
    this.perplexitySlider.value = this.perplexity.toString();
    this.perplexitySlider.addEventListener('change', () =>
      this.updateTSNEPerplexityFromSliderChange()
    );
    this.updateTSNEPerplexityFromSliderChange();
    this.learningRateInput.addEventListener('change', () =>
      this.updateTSNELearningRateFromUIChange()
    );
    this.updateTSNELearningRateFromUIChange();
    this.superviseFactorInput.addEventListener('change', () =>
      this.updateTSNESuperviseFactorFromUIChange()
    );
    this.updateTSNESuperviseFactorFromUIChange();*/
    this.setupCustomProjectionInputFields();
    // TODO: figure out why `--paper-input-container-input` css mixin didn't
    // work.
    const inputs = this.root.querySelectorAll(
      'paper-dropdown-menu paper-input input'
    );
    for (let i = 0; i < inputs.length; i++) {
      (inputs[i] as HTMLElement).style.fontSize = '14px';
    }
  }
  restoreUIFromBookmark(bookmark: State) {
    this.disablePolymerChangesTriggerReprojection();
    // PCA
    this.pcaX = bookmark.pcaComponentDimensions[0];
    this.pcaY = bookmark.pcaComponentDimensions[1];
    if (bookmark.pcaComponentDimensions.length === 3) {
      this.pcaZ = bookmark.pcaComponentDimensions[2];
    }
    this.pcaIs3d = bookmark.pcaComponentDimensions.length === 3;
    // t-SNE
    /*
    if (this.perplexitySlider) {
      this.perplexitySlider.value = bookmark.tSNEPerplexity.toString();
    }
    if (this.learningRateInput) {
      this.learningRateInput.value = bookmark.tSNELearningRate.toString();
    }*/
    //this.tSNEis3d = bookmark.tSNEis3d;
    // UMAP
    this.umapIs3d = bookmark.umapIs3d;
    this.umapNeighbors = bookmark.umapNeighbors;
    // custom
    this.customSelectedSearchByMetadataOption =
      bookmark.customSelectedSearchByMetadataOption;
    if (this.customProjectionXLeftInput) {
      this.customProjectionXLeftInput.set(
        bookmark.customXLeftText,
        bookmark.customXLeftRegex
      );
    }
    if (this.customProjectionXRightInput) {
      this.customProjectionXRightInput.set(
        bookmark.customXRightText,
        bookmark.customXRightRegex
      );
    }
    if (this.customProjectionYUpInput) {
      this.customProjectionYUpInput.set(
        bookmark.customYUpText,
        bookmark.customYUpRegex
      );
    }
    if (this.customProjectionYDownInput) {
      this.customProjectionYDownInput.set(
        bookmark.customYDownText,
        bookmark.customYDownRegex
      );
    }
    this.computeAllCentroids();
    this.setZDropdownEnabled(this.pcaIs3d);
    //this.updateTSNEPerplexityFromSliderChange();
    //this.updateTSNELearningRateFromUIChange();
    if (this.iterationLabelTsne) {
      this.iterationLabelTsne.innerText = bookmark.tSNEIteration.toString();
    }
    if (bookmark.selectedProjection != null) {
      this.showTab(bookmark.selectedProjection);
    }
    this.enablePolymerChangesTriggerReprojection();
  }
  populateBookmarkFromUI(bookmark: State) {
    this.disablePolymerChangesTriggerReprojection();
    // PCA
    bookmark.pcaComponentDimensions = [this.pcaX, this.pcaY];
    if (this.pcaIs3d) {
      bookmark.pcaComponentDimensions.push(this.pcaZ);
    }
    // t-SNE
    /*
    if (this.perplexitySlider != null) {
      bookmark.tSNEPerplexity = +this.perplexitySlider.value;
    }
    if (this.learningRateInput != null) {
      bookmark.tSNELearningRate = +this.learningRateInput.value;
    }*/
    //bookmark.tSNEis3d = this.tSNEis3d;
    // UMAP
    bookmark.umapIs3d = this.umapIs3d;
    // custom
    bookmark.customSelectedSearchByMetadataOption = this.customSelectedSearchByMetadataOption;
    if (this.customProjectionXLeftInput != null) {
      bookmark.customXLeftText = this.customProjectionXLeftInput.getValue();
      bookmark.customXLeftRegex = this.customProjectionXLeftInput.getInRegexMode();
    }
    if (this.customProjectionXRightInput != null) {
      bookmark.customXRightText = this.customProjectionXRightInput.getValue();
      bookmark.customXRightRegex = this.customProjectionXRightInput.getInRegexMode();
    }
    if (this.customProjectionYUpInput != null) {
      bookmark.customYUpText = this.customProjectionYUpInput.getValue();
      bookmark.customYUpRegex = this.customProjectionYUpInput.getInRegexMode();
    }
    if (this.customProjectionYDownInput != null) {
      bookmark.customYDownText = this.customProjectionYDownInput.getValue();
      bookmark.customYDownRegex = this.customProjectionYDownInput.getInRegexMode();
    }
    this.enablePolymerChangesTriggerReprojection();
  }
  // This method is marked as public as it is used as the view method that
  // abstracts DOM manipulation so we can stub it in a test.
  // TODO(nsthorat): Move this to its own class as the glue between this class
  // and the DOM.
  setZDropdownEnabled(enabled: boolean) {
    if (this.zDropdown) {
      if (enabled) {
        this.zDropdown.removeAttribute('disabled');
      } else {
        this.zDropdown.setAttribute('disabled', 'true');
      }
    }
  }
  dataSetUpdated(dataSet: DataSet, originalDataSet: DataSet, dim: number) {
    this.dataSet = dataSet;
    this.originalDataSet = originalDataSet;
    this.dim = dim;
    const pointCount = dataSet == null ? 0 : dataSet.points.length;
    //const perplexity = Math.max(5, Math.ceil(Math.sqrt(pointCount) / 4));
    //this.perplexitySlider.value = perplexity.toString();
    //this.updateTSNEPerplexityFromSliderChange();
    this.clearCentroids();
    (this.$$('#tsne-sampling') as HTMLElement).style.display =
      pointCount > TSNE_SAMPLE_SIZE ? null : 'none';
    const wasSampled =
      dataSet == null
        ? false
        : dataSet.dim[0] > PCA_SAMPLE_DIM || dataSet.dim[1] > PCA_SAMPLE_DIM;
    (this.$$('#pca-sampling') as HTMLElement).style.display = wasSampled
      ? null
      : 'none';
    this.showTab('tsne');
  }
  @observe('pcaIs3d')
  _pcaDimensionToggleObserver() {
    this.setZDropdownEnabled(this.pcaIs3d);
    this.beginProjection(this.currentProjection);
  }

  @observe('umapIs3d')
  _umapDimensionToggleObserver() {
    this.beginProjection(this.currentProjection);
  }

  @observe('temporalStatus')
  _DVITemporalStatusObserver(){

  }
  metadataChanged(spriteAndMetadata: SpriteAndMetadataInfo) {
    // Project by options for custom projections.
    let searchByMetadataIndex = -1;
    this.searchByMetadataOptions = spriteAndMetadata.stats.map((stats, i) => {
      // Make the default label by the first non-numeric column.
      if (!stats.isNumeric && searchByMetadataIndex === -1) {
        searchByMetadataIndex = i;
      }
      return stats.name;
    });
    this.customSelectedSearchByMetadataOption = this.searchByMetadataOptions[
      Math.max(0, searchByMetadataIndex)
    ];
  }
  public showTab(id: ProjectionType) {
    this.currentProjection = id;
    const tab = this.$$('.ink-tab[data-tab="' + id + '"]') as HTMLElement;
    const allTabs = this.root.querySelectorAll('.ink-tab');
    for (let i = 0; i < allTabs.length; i++) {
      util.classed(allTabs[i] as HTMLElement, 'active', false);
    }
    util.classed(tab, 'active', true);
    const allTabContent = this.root.querySelectorAll('.ink-panel-content');
    for (let i = 0; i < allTabContent.length; i++) {
      util.classed(allTabContent[i] as HTMLElement, 'active', false);
    }
    util.classed(
      this.$$('.ink-panel-content[data-panel="' + id + '"]') as HTMLElement,
      'active',
      true
    );
    // guard for unit tests, where polymer isn't attached and $ doesn't exist.
    if (this.$ != null) {
      const main = this.$['main'];
      // In order for the projections panel to animate its height, we need to
      // set it explicitly.
      requestAnimationFrame(() => {
        this.style.height = main.clientHeight + 'px';
      });
    }
    console.log(id);
    this.beginProjection(id);
  }
  private beginProjection(projection: ProjectionType) {
    if (this.polymerChangesTriggerReprojection === false) {
      return;
    }
    if (projection === 'pca') {
      if (this.dataSet != null) {
        this.dataSet.stopTSNE();
      }
      this.showPCA();
    } else if (projection === 'tsne') {
      this.showTSNE();
    } else if (projection === 'umap') {
      this.showUmap();
    } else if (projection === 'custom') {
      if (this.dataSet != null) {
        this.dataSet.stopTSNE();
      }
      this.computeAllCentroids();
      this.reprojectCustom();
    }
  }
  private showTSNE() {
    const dataSet = this.dataSet;
    if (dataSet == null) {
      return;
    }
    const accessors = getProjectionComponents('tsne', [
      0,
      1,
      this.tSNEis3d ? 2 : null,
    ]);
    const dimensionality = this.tSNEis3d ? 3 : 2;
    const projection = new Projection(
      'tsne',
      accessors,
      dimensionality,
      dataSet
    );
    this.projector.setProjection(projection);
    if (this.dataSet.hasTSNERun) {
      this.projector.notifyProjectionPositionsUpdated();
    }
  }
  private runTSNE() {
    let projectionChangeNotified = false;
    this.runTsneButton.innerText = 'Stop';
    this.runTsneButton.disabled = true;
    this.pauseTsneButton.innerText = 'Pause';
    this.pauseTsneButton.disabled = true;
    this.previousDVIButton.disabled = true;
    this.nextDVIButton.disabled = true;
    //this.perturbTsneButton.disabled = false;
    this.dataSet.projectTSNE(
      this.perplexity,
      this.learningRate,
      this.tSNEis3d ? 3 : 2,
      (iteration: number, dataset?: DataSet, totalIter?: number) => {
        if (iteration != null) {
          this.runTsneButton.disabled = false;
          this.pauseTsneButton.disabled = false;
          this.iterationLabelTsne.innerText = '' + iteration;
          this.totalIterationLabelDVI.innerText = '' + totalIter;
          this.projector.notifyProjectionPositionsUpdated();
          if (!projectionChangeNotified && this.dataSet.projections['tsne']) {
            this.projector.onProjectionChanged();
            projectionChangeNotified = true;
          }
        } else {
          this.runTsneButton.innerText = 'Re-run';
          this.runTsneButton.disabled = false;
          this.pauseTsneButton.innerText = 'Pause';
          this.pauseTsneButton.disabled = true;
          //this.perturbTsneButton.disabled = true;
          this.previousDVIButton.disabled = false;
          this.projector.onProjectionChanged();
        }
      }
    );
  }
  private showUmap() {
    const dataSet = this.dataSet;
    if (dataSet == null) {
      return;
    }
    const accessors = getProjectionComponents('umap', [
      0,
      1,
      this.umapIs3d ? 2 : null,
    ]);
    const dimensionality = this.umapIs3d ? 3 : 2;
    const projection = new Projection(
      'umap',
      accessors,
      dimensionality,
      dataSet
    );
    this.projector.setProjection(projection);
    if (!this.dataSet.hasUmapRun) {
      this.runUmap();
    } else {
      this.projector.notifyProjectionPositionsUpdated();
    }
  }
  private runUmap() {
    let projectionChangeNotified = false;
    this.runUmapButton.disabled = true;
    const nComponents = this.umapIs3d ? 3 : 2;
    const nNeighbors = this.umapNeighbors;
    this.dataSet.projectUmap(nComponents, nNeighbors, (iteration: number, bg: string) => {
      if (iteration != null) {
        this.runUmapButton.disabled = false;
        this.projector.notifyProjectionPositionsUpdated();
        if (!projectionChangeNotified && this.dataSet.projections['umap']) {
          this.projector.onProjectionChanged();
          projectionChangeNotified = true;
        }
      } else {
        this.runUmapButton.innerText = 'Re-run';
        this.runUmapButton.disabled = false;
        this.projector.onProjectionChanged();
      }
    });
  }
  @observe('pcaX', 'pcaY', 'pcaZ')
  private showPCAIfEnabled() {
    if (this.polymerChangesTriggerReprojection) {
      this.showPCA();
    }
  }
  private updateTotalVarianceMessage() {
    let variances = this.dataSet.fracVariancesExplained;
    let totalVariance = variances[this.pcaX] + variances[this.pcaY];
    let msg = 'Total variance described: ';
    if (this.pcaIs3d) {
      totalVariance += variances[this.pcaZ];
    }
    msg += (totalVariance * 100).toFixed(1) + '%.';
    (this.$$('#total-variance') as HTMLElement).textContent = msg;
  }
  private showPCA() {
    if (this.dataSet == null) {
      return;
    }
    this.dataSet.projectPCA().then(() => {
      // Polymer properties are 1-based.
      const accessors = getProjectionComponents('pca', [
        this.pcaX,
        this.pcaY,
        this.pcaZ,
      ]);
      const dimensionality = this.pcaIs3d ? 3 : 2;
      const projection = new Projection(
        'pca',
        accessors,
        dimensionality,
        this.dataSet
      );
      this.projector.setProjection(projection);
      let numComponents = Math.min(NUM_PCA_COMPONENTS, this.dataSet.dim[1]);
      this.updateTotalVarianceMessage();
      this.pcaComponents = util.range(numComponents).map((i) => {
        let fracVariance = this.dataSet.fracVariancesExplained[i];
        return {
          id: i,
          componentNumber: i + 1,
          percVariance: (fracVariance * 100).toFixed(1),
        };
      });
    });
  }
  private reprojectCustom() {
    if (
      this.centroids == null ||
      this.centroids.xLeft == null ||
      this.centroids.xRight == null ||
      this.centroids.yUp == null ||
      this.centroids.yDown == null
    ) {
      return;
    }
    const xDir = vector.sub(this.centroids.xRight, this.centroids.xLeft);
    this.dataSet.projectLinear(xDir, 'linear-x');
    const yDir = vector.sub(this.centroids.yUp, this.centroids.yDown);
    this.dataSet.projectLinear(yDir, 'linear-y');
    const accessors = getProjectionComponents('custom', ['x', 'y']);
    const projection = new Projection('custom', accessors, 2, this.dataSet);
    this.projector.setProjection(projection);
  }
  clearCentroids(): void {
    this.centroids = {xLeft: null, xRight: null, yUp: null, yDown: null};
    this.allCentroid = null;
  }
  @observe('customSelectedSearchByMetadataOption')
  _customSelectedSearchByMetadataOptionChanged(newVal: string, oldVal: string) {
    if (this.polymerChangesTriggerReprojection === false) {
      return;
    }
    if (this.currentProjection === 'custom') {
      this.computeAllCentroids();
      this.reprojectCustom();
    }
  }
  private setupCustomProjectionInputFields() {
    this.customProjectionXLeftInput = this.setupCustomProjectionInputField(
      'xLeft'
    );
    this.customProjectionXRightInput = this.setupCustomProjectionInputField(
      'xRight'
    );
    this.customProjectionYUpInput = this.setupCustomProjectionInputField('yUp');
    this.customProjectionYDownInput = this.setupCustomProjectionInputField(
      'yDown'
    );
  }
  private computeAllCentroids() {
    this.computeCentroid('xLeft');
    this.computeCentroid('xRight');
    this.computeCentroid('yUp');
    this.computeCentroid('yDown');
  }
  private computeCentroid(name: InputControlName) {
    const input = this.$$('#' + name) as any;
    if (input == null) {
      return;
    }
    const value = input.getValue();
    if (value == null) {
      return;
    }
    let inRegexMode = input.getInRegexMode();
    let result = this.getCentroid(value, inRegexMode);
    if (result.numMatches === 0) {
      input.message = '0 matches. Using a random vector.';
      result.centroid = vector.rn(this.dim);
    } else {
      input.message = `${result.numMatches} matches.`;
    }
    this.centroids[name] = result.centroid;
    this.centroidValues[name] = value;
  }
  private setupCustomProjectionInputField(name: InputControlName): any {
    let input = this.$$('#' + name) as any;
    input.registerInputChangedListener((input, inRegexMode) => {
      if (this.polymerChangesTriggerReprojection) {
        this.computeCentroid(name);
        this.reprojectCustom();
      }
    });
    return input;
  }
  private getCentroid(pattern: string, inRegexMode: boolean): CentroidResult {
    if (pattern == null || pattern === '') {
      return {numMatches: 0};
    }
    // Search by the original dataset since we often want to filter and project
    // only the nearest neighbors of A onto B-C where B and C are not nearest
    // neighbors of A.
    let accessor = (i: number) => this.originalDataSet.points[i].vector;
    let result = this.originalDataSet.query(
      pattern,
      inRegexMode,
      this.customSelectedSearchByMetadataOption
    );
    let r = result[1];
    return {centroid: vector.centroid(r, accessor), numMatches: r.length};
  }
  getPcaSampledDimText() {
    return PCA_SAMPLE_DIM.toLocaleString();
  }
  getPcaSampleSizeText() {
    return PCA_SAMPLE_SIZE.toLocaleString();
  }
  getTsneSampleSizeText() {
    return TSNE_SAMPLE_SIZE.toLocaleString();
  }
  getUmapSampleSizeText() {
    return UMAP_SAMPLE_SIZE.toLocaleString();
  }

}
