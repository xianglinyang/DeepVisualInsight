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

import {html} from '@polymer/polymer';

import './styles';

export const template = html`
    <style include="vz-projector-styles"></style>
    <style>
      :host {
        transition: height 0.2s;
      }

      .ink-button {
        border: none;
        border-radius: 2px;
        font-size: 13px;
        padding: 10px;
        min-width: 80px;
        flex-shrink: 0;
        background: #e3e3e3;
      }

      .ink-panel-buttons {
        margin-bottom: 10px;
      }

      .two-way-toggle {
        display: flex;
        flex-direction: row;
      }

      .two-way-toggle span {
        padding-right: 15px;
      }

      .has-border {
        border: 1px solid rgba(0, 0, 0, 0.1);
      }

      .toggle {
        min-width: 0px;
        font-size: 12px;
        width: 17px;
        min-height: 0px;
        height: 21px;
        padding: 0;
        margin: 0px;
      }

      .toggle[active] {
        background-color: #880e4f;
        color: white;
      }

      .two-columns {
        display: flex;
        justify-content: space-between;
      }

      .two-columns > :first-child {
        margin-right: 15px;
      }

      .two-columns > div {
        width: 50%;
      }
      
      .two-rows {
        display: flex;
        justify-content: space-between;
        flex-direction: column;
      }
      
      .row {
        display: flex;
        justify-content: space-between;
      }

      .dropdown-item {
        justify-content: space-between;
        min-height: 35px;
      }

      .tsne-supervise-factor {
        margin-bottom: -8px;
      }

      #z-container {
        display: flex;
        align-items: center;
        width: 50%;
      }

      #z-checkbox {
        margin: 27px 0 0 5px;
        width: 18px;
      }

      #z-dropdown {
        flex-grow: 1;
      }

      .notice {
        color: #880e4f;
      }

      .container {
        padding: 20px;
      }

      .book-icon {
        height: 20px;
        color: rgba(0, 0, 0, 0.7);
      }

      .item-details {
        color: gray;
        font-size: 12px;
        margin-left: 5px;
      }

      .pca-dropdown {
        width: 100%;
      }

      .pca-dropdown paper-listbox {
        width: 135px;
      }

      .dropdown-item.header {
        border-bottom: 1px solid #aaa;
        color: #333;
        font-weight: bold;
      }

      #total-variance {
        color: rgba(0, 0, 0, 0.7);
      }
      table, th, td {
        border: 1px solid black;
        padding: 15px;
        border-collapse: collapse;
      }
      
    </style>
    <div id="main">
      <div class="ink-panel-header">
        <div class="ink-tab-group">

          <div data-tab="tsne" id="tsne-tab" class="ink-tab projection-tab">
            DVI
          </div>
          <paper-tooltip
            for="tsne-tab"
            position="bottom"
            animation-delay="0"
            fit-to-visible-bounds
          >
            Deep Visual Insight
          </paper-tooltip>

        </div>
      </div>
      <div class="container">
        <!-- UMAP Controls -->
        <div data-panel="umap" class="ink-panel-content">
          <div class="slider">
            <label>Dimension</label>
            <div class="two-way-toggle">
              <span>2D</span>
              <paper-toggle-button id="umap-toggle" checked="{{umapIs3d}}"
                >3D</paper-toggle-button
              >
            </div>
          </div>
          <div class="slider umap-neighbors">
            <label>
              Neighbors
              <paper-icon-button
                icon="help"
                class="help-icon"
              ></paper-icon-button>
              <paper-tooltip
                position="right"
                animation-delay="0"
                fit-to-visible-bounds
              >
                The number of nearest neighbors used to compute the fuzzy
                simplicial set, which is used to approximate the overall shape
                of the manifold. The default value is 15.
              </paper-tooltip>
            </label>
            <paper-slider
              id="umap-neighbors-slider"
              value="{{umapNeighbors}}"
              pin
              min="5"
              max="50"
            ></paper-slider>
            <span>[[umapNeighbors]]</span>
          </div>
          <p>
            <button
              id="run-umap"
              class="ink-button"
              title="Run UMAP"
              on-tap="runUmap"
            >
              Run
            </button>
          </p>
          <p id="umap-sampling" class="notice">
            For faster results, the data will be sampled down to
            [[getUmapSampleSizeText()]] points.
          </p>
          <p>
            <iron-icon icon="book" class="book-icon"></iron-icon>
            <a
              target="_blank"
              rel="noopener"
              href="https://umap-learn.readthedocs.io/en/latest/how_umap_works.html"
            >
              Learn more about UMAP.
            </a>
          </p>
        </div>
        <!-- TSNE Controls -->
        <div data-panel="tsne" class="ink-panel-content">
          <!-- Subject Model Path -->
        <div class="subject-model-path-editor">
            <paper-input
              value="{{subjectModelPathEditorInput}}"
              label="Model Path"
              on-input="subjectModelPathEditorInputChange"
            >
            </paper-input>
    </div>
    <!-- Misc Setting -->
    <div class="misc-setting-editor">
        </paper-input>
        <paper-input
          value="{{resolutionEditorInput}}"
          label="Resolution"
          on-input="resolutionEditorInputChange"
        >
        </paper-input>
    </div>
        <div class="slider">
            <label>Status</label>
            <div class="two-way-toggle">
              <span>Indices</span>
              <paper-toggle-button id="DVI-toggle" checked="{{temporalStatus}}">
                  Search Predicates
              </paper-toggle-button>
            </div>
          </div>
          <!--
           <div class="two-rows">
              <div class="row">
                <button class="run-tsne ink-button" title="Re-run DVI">
                  Run
                </button>
                <button class="pause-tsne ink-button" title="Pause DVI">
                  Pause
                </button>
              </div> 
              <div class="row">
                 <button class="previous-dvi ink-button" title="Previous DVI">
                   Previous
                 </button>
                 <button class="next-dvi ink-button" title="Next DVI">
                   Next
                 </button>
              </div>
          </div> -->
          <div class="row">
            <button class="load-dvi ink-button" title="Load DVI">
              Load
            </button>
            <button class="previous-dvi ink-button" title="Previous DVI">
              Previous
            </button>
            <button class="next-dvi ink-button" title="Next DVI">
              Next
            </button>
          </div>
            <div class="row">  -- </div>
            
            <div class="row">
                <div class="iteration-editor">
                    <paper-input
                      value="{{iterationEditorInput}}"
                      label="Iteration"
                      on-input="iterationEditorInputChange"
                    ></paper-input>
                </div>
                <button class="jump-dvi ink-button" title="Jump DVI">Jump</button>
            </div>
          <p>Iteration: <span class="run-tsne-iter">0</span></p>
          <div>
              <table>
                  <caption>Visualization Confidence</caption>
                <tr>
                  <td></td>
                  <td>train</td>
                  <td>test</td>
                </tr>
                <tr>
                  <td>nn</td>
                  <td><span class="nn_train_15">NA</span> </td>
                  <td><span class="nn_test_15">NA</span></td>
                </tr>
                  <tr>
                      <td>boundary</td>
                      <td><span class="bound_train_15">NA</span></td>
                      <td><span class="bound_test_15">NA</span></td>
                  </tr>
                <tr>
                  <td>PPR</td>
                  <td><span class="inv_acc_train">NA</span> </td>
                  <td> <span class="inv_acc_test">NA</span></td>
                </tr>
                <tr>
                  <td>CCR</td>
                  <td><span class="inv_conf_train">NA</span></td>
                  <td><span class="inv_conf_test">NA</span></td>
                </tr>
              </table>
<!--          <p>Projection nn perseverance knn: (train,15): <span class="nn_train_15">NA</span> (test,15): <span class="nn_test_15">NA</span></p>-->
<!--          <p>Projection boundary perserverance knn: (train,15): <span class="bound_train_15">NA</span> (test,15): <span class="bound_test_15">NA</span></p>-->
<!--          <p>PPR: train: <span class="inv_acc_train">NA</span> test: <span class="inv_acc_test">NA</span></p>-->
<!--          <p>CCR: train: <span class="inv_conf_train">NA</span> test: <span class="inv_conf_test">NA</span></p>-->
          <p>Task Model Accuracy:</p>
          <p>train: <span class="acc_train">NA</span> test: <span class="acc_test">NA</span></p>
          </div>
           <p>Total iteration number: <span class="dvi-total-iter">0</span></p>
          <p id="tsne-sampling" class="notice">
          </p>
        </div>
        <!-- PCA Controls -->
        <div data-panel="pca" class="ink-panel-content">
          <div class="two-columns">
            <div>
              <!-- Left column -->
              <paper-dropdown-menu
                class="pca-dropdown"
                vertical-align="bottom"
                no-animations
                label="X"
              >
                <paper-listbox
                  attr-for-selected="value"
                  class="dropdown-content"
                  selected="{{pcaX}}"
                  slot="dropdown-content"
                >
                  <paper-item disabled class="dropdown-item header">
                    <div>#</div>
                    <div>Variance (%)</div>
                  </paper-item>
                  <template is="dom-repeat" items="[[pcaComponents]]">
                    <paper-item
                      class="dropdown-item"
                      value="[[item.id]]"
                      label="Component #[[item.componentNumber]]"
                    >
                      <div>[[item.componentNumber]]</div>
                      <div class="item-details">[[item.percVariance]]</div>
                    </paper-item>
                  </template>
                </paper-listbox>
              </paper-dropdown-menu>
              <paper-dropdown-menu
                class="pca-dropdown"
                no-animations
                vertical-align="bottom"
                label="Z"
                disabled="[[!hasPcaZ]]"
                id="z-dropdown"
              >
                <paper-listbox
                  attr-for-selected="value"
                  class="dropdown-content"
                  selected="{{pcaZ}}"
                  slot="dropdown-content"
                >
                  <paper-item disabled class="dropdown-item header">
                    <div>#</div>
                    <div>Variance (%)</div>
                  </paper-item>
                  <template is="dom-repeat" items="[[pcaComponents]]">
                    <paper-item
                      class="dropdown-item"
                      value="[[item.id]]"
                      label="Component #[[item.componentNumber]]"
                    >
                      <div>[[item.componentNumber]]</div>
                      <div class="item-details">[[item.percVariance]]</div>
                    </paper-item>
                  </template>
                </paper-listbox>
              </paper-dropdown-menu>
            </div>
            <div>
              <!-- Right column -->
              <paper-dropdown-menu
                class="pca-dropdown"
                vertical-align="bottom"
                no-animations
                label="Y"
              >
                <paper-listbox
                  attr-for-selected="value"
                  class="dropdown-content"
                  selected="{{pcaY}}"
                  slot="dropdown-content"
                >
                  <paper-item disabled class="dropdown-item header">
                    <div>#</div>
                    <div>Variance (%)</div>
                  </paper-item>
                  <template is="dom-repeat" items="[[pcaComponents]]">
                    <paper-item
                      class="dropdown-item"
                      value="[[item.id]]"
                      label="Component #[[item.componentNumber]]"
                    >
                      <div>[[item.componentNumber]]</div>
                      <div class="item-details">[[item.percVariance]]</div>
                    </paper-item>
                  </template>
                </paper-listbox>
              </paper-dropdown-menu>
              <paper-checkbox
                id="z-checkbox"
                checked="{{pcaIs3d}}"
              ></paper-checkbox>
            </div>
          </div>
          <p id="pca-sampling" class="notice">
            PCA is approximate.
            <paper-icon-button
              icon="help"
              class="help-icon"
            ></paper-icon-button>
          </p>
          <div id="total-variance">Total variance</div>
          <paper-tooltip
            for="pca-sampling"
            position="top"
            animation-delay="0"
            fit-to-visible-bounds
          >
            For fast results, the data was sampled to [[getPcaSampleSizeText()]]
            points and randomly projected down to [[getPcaSampledDimText()]]
            dimensions.
          </paper-tooltip>
        </div>
        <!-- Custom Controls -->
        <div data-panel="custom" class="ink-panel-content">
          <paper-dropdown-menu
            style="width: 100%"
            no-animations
            label="Search by"
          >
            <paper-listbox
              attr-for-selected="value"
              class="dropdown-content"
              selected="{{customSelectedSearchByMetadataOption}}"
              slot="dropdown-content"
            >
              <template is="dom-repeat" items="[[searchByMetadataOptions]]">
                <paper-item
                  class="dropdown-item"
                  value="[[item]]"
                  label="[[item]]"
                >
                  [[item]]
                </paper-item>
              </template>
            </paper-listbox>
          </paper-dropdown-menu>
          <div class="two-columns">
            <vz-projector-input id="xLeft" label="Left"></vz-projector-input>
            <vz-projector-input id="xRight" label="Right"></vz-projector-input>
          </div>
          <div class="two-columns">
            <vz-projector-input id="yUp" label="Up"></vz-projector-input>
            <vz-projector-input id="yDown" label="Down"></vz-projector-input>
          </div>
        </div>
      </div>
    </div>
  </template>
  <script src="vz-projector-projections-panel.js"></script>
</dom-module>
`;
