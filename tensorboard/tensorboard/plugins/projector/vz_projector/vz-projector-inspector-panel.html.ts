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

export const template = html`
  <style include="vz-projector-styles"></style>
  <style>
    :host {
      display: flex;
      flex-direction: column;
      /* Account for the bookmark pane at the bottom */
      height: calc(100% - 55px);
    }

    .container {
      display: block;
      padding: 10px 20px 0 20px;
    }

    .buttons {
      display: flex;
      height: 60px;
    }

    .button {
      margin-right: 10px;
      border: none;
      border-radius: 7px;
      font-size: 13px;
      padding: 10px;
      background: #e3e3e3;
    }

    .button:last-child {
      margin-right: 0;
    }
    
    .search-button{
        display: flex;
        margin-right: 10px;
        width: 60px;
        height: 30px;
        font-size: 13px;
    }
    .boundingbox-button{
        display: flex;
        margin-right: 10px;
        width: 60px;
        height: 30px;
        font-size: 13px;
    }


    .nn,
    .metadata-info {
      display: flex;
      flex-direction: column;
    }

    .nn > *,
    .metadata-info > * {
      padding: 0 20px;
    }

    .nn-list,
    .metadata-list {
      overflow-y: auto;
    }

    .nn-list .neighbor,
    .metadata-list .metadata {
      font-size: 12px;
      margin-bottom: 8px;
    }

    .nn-list .label-and-value,
    .metadata-list .label-and-value {
      display: flex;
      justify-content: space-between;
    }

    .label {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .nn-list .value,
    .metadata-list .value {
      color: #666;
      float: right;
      font-weight: 300;
      margin-left: 8px;
    }

    .nn-list .bar,
    .metadata-list .bar {
      position: relative;
      border-top: 1px solid rgba(0, 0, 0, 0.15);
      margin: 2px 0;
    }

    .nn-list .bar .fill,
    .metadata-list .bar .fill {
      position: absolute;
      top: -1px;
      border-top: 1px solid white;
    }

    .nn-list .tick,
    .metadata-list .tick {
      position: absolute;
      top: 0px;
      height: 3px;
      border-left: 1px solid rgba(0, 0, 0, 0.15);
    }

    .nn-list .sprite-image,
    .metadata-list .sprite-image {
      width: 100%;
    }

    .nn-list.nn-img-show .sprite-image,
    .metadata-list.nn-img-show .sprite-image {
      display: block;
    }

    .nn-list .neighbor-link:hover,
    .metadata-list .metadata-link:hover {
      cursor: pointer;
    }

    .search-by {
      display: flex;
    }

    .search-by vz-projector-input {
      width: 100%;
    }

    .search-by paper-dropdown-menu {
      margin-left: 10px;
      width: 120px;
    }
    
    .search-by button {
        margin-right: 10px;
        width: 60px;
    }

    .distance .options {
      float: right;
    }

    .neighbor-image-controls {
      display: flex;
      padding: 0.8em 0.1em;
    }

    .options a {
      color: #727272;
      font-size: 13px;
      margin-left: 12px;
      text-decoration: none;
    }

    .options a.selected {
      color: #009efe;
    }

    .neighbors {
      margin-bottom: 15px;
    }

    .neighbors-options {
      margin-top: 6px;
    }

    .neighbors-options .option-label,
    .distance .option-label {
      color: #727272;
      margin-right: 2px;
      width: auto;
    }

    .num-neighbors-container {
      display: inline-block;
    }

    .nn-slider {
      --paper-slider-input: {
        width: 64px;
      }
      --paper-input-container-input: {
        font-size: 14px;
      }
    }

    .euclidean {
      margin-right: 10px;
    }

    .matches-list {
      padding: 0 20px;
    }

    .matches-list .row {
      border-bottom: 1px solid #ddd;
      cursor: pointer;
      display: flex;
      font-size: 12px;
      margin: 5px 0;
      padding: 4px 0;
    }

    .results {
      display: flex;
      flex-direction: column;
    }

    .results,
    .nn,
    .nn-list {
      flex: 1 0 100px;
    }
    
    [hidden] {
         display: none;
      }
  </style>
  <div class="container">
    <div class="buttons">
      <button class="button reset-filter">Show All Data</button>
      <button class="button set-filter">Filter selection</button>
      <button class="button clear-selection">Clear selection</button>
    </div>
    
    <div class="search-by" style="margin-top:20px">
        <vz-projector-input id="search-box" label="High level query"></vz-projector-input>
        <button class="search-button search">query</button>
        <div>
        </div>
    </div>
    <div class="query-list" style="display: none">
      <div class="list"></div>
    </div>

    <div>
    </div>
  </div>
  <div style="display: none">
      <button class="boundingbox-button add">add</button>
      <button class="boundingbox-button reset">reset</button>
      <button class="boundingbox-button sent">sent</button>
      <button class="boundingbox-button show">show</button>
      <p>Current <span class="boundingBoxSelection">NA</span></p>
  </div>

  <div class="results">
    <div class="nn" style="display: none">
      <div class="neighbors">
        <div class="neighbors-options">
          <div hidden$="[[!noShow]]" class="slider num-nn">
            <span class="option-label">neighbors</span>
            <paper-icon-button
              icon="help"
              class="help-icon"
            ></paper-icon-button>
            <paper-tooltip
              position="bottom"
              animation-delay="0"
              fit-to-visible-bounds
            >
              The number of neighbors (in the original space) to show when
              clicking on a point.
            </paper-tooltip>
            <paper-slider
              class="nn-slider"
              pin
              min="5"
              max="999"
              editable
              value="{{numNN}}"
              on-change="updateNumNN"
            ></paper-slider>
          </div>
        </div>
        <div hidden$="[[!noShow]]" class="distance">
          <span class="option-label">distance</span>
          <div class="options">
            <a class="selected cosine" href="javascript:void(0);">COSINE</a>
            <a class="euclidean" href="javascript:void(0);">EUCLIDEAN</a>
          </div>
        </div>
        <div class="neighbor-image-controls">
          <template is="dom-if" if="[[spriteImagesAvailable]]">
            <paper-checkbox checked="{{showNeighborImages}}">
              show images
              <paper-icon-button
                icon="help"
                class="help-icon"
              ></paper-icon-button>
              <paper-tooltip
                position="bottom"
                animation-delay="0"
                fit-to-visible-bounds
              >
                Show the original images of the point.
              </paper-tooltip>
            </paper-checkbox>
          </template>
        </div>
      </div>
      <div class="nn-list"></div>
    </div>
    <div class="metadata-info" style="display: none">
      <div class="neighbors-options">
        <div class="slider num-nn">
          <span class="option-label">neighbors</span>
          <paper-icon-button icon="help" class="help-icon"></paper-icon-button>
          <paper-tooltip
            position="bottom"
            animation-delay="0"
            fit-to-visible-bounds
          >
            The number of neighbors (in the selected space) to show when
            clicking on a point.
          </paper-tooltip>
          <paper-slider
            class="nn-slider"
            pin
            min="5"
            max="999"
            editable
            value="{{numNN}}"
            on-change="updateNumNN"
          ></paper-slider>
        </div>
      </div>
      <p>{{metadataColumn}} labels (click to apply):</p>
      <div class="metadata-list"></div>
    </div>
    <div class="matches-list" style="display: none">
      <div class="list"></div>
      <div class="limit-msg">Showing only the first 100 results...</div>
    </div>
  </div>
`;
