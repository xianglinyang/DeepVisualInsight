from flask import request, Flask, jsonify, make_response
from flask_cors import CORS, cross_origin

import os, sys
import numpy as np
import base64
import json
import torch
import math

lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from prepare_data import prepare_data

# flask for API server
app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/animation', methods=["POST"])
@cross_origin()
def animation():
    res = request.get_json()
    path = res['path']
    data_path = res['data_path']
    iteration = res['iteration']
    cache = res['cache']
    resolution = int(res['resolution'])
    folder_path = 'data/' + path.split('/')[-1]
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    path_files = os.listdir(path+'/Model')
    maximum_iteration = len(path_files) - 2
    data = torch.load(data_path + "/testing_dataset_data.pth")
    labels = torch.load(data_path + "/testing_dataset_label.pth").tolist()
    if not res['cache']:
        prepare_data(path, data, iteration, resolution, folder_path, False)

    with open(folder_path+'/dimension_reduction_'+str(iteration)+'.npy', 'rb') as f:
        result = np.load(f).tolist()

    with open(folder_path+'/grid_'+str(iteration)+'.npy', 'rb') as f:
        grid_index = np.load(f)
        grid_index = grid_index.reshape((-1, 2)).tolist()
    with open(folder_path+'/decision_view_'+str(iteration)+'.npy', 'rb') as f:
        grid_color = np.load(f)
        grid_color = grid_color.reshape((-1, 3))
        grid_color *= 255
        grid_color = grid_color.astype(int).tolist()

    with open(folder_path+'/color.npy', 'rb') as f:
        standard_color = np.load(f)*255
        standard_color = standard_color.astype(int).tolist()

    label_color_list = []
    for label in labels:
        label_color_list.append(standard_color[label])

    return make_response(jsonify({'result': result, 'grid_index': grid_index, 'grid_color': grid_color, 'label_color_list':label_color_list, 'maximum_iteration':maximum_iteration}), 200)

@app.route('/', methods=["POST", "GET"])
def index():
    return 'Index Page'


@app.route('/hello')
def hello_world():
    return 'Hello World!'


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":

    app.run(host="192.168.254.128")
