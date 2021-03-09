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
    raw_data = []
    for data in res['rawdata']:
        vector = []
        for dimension in data:
            vector.append(data[dimension])
        raw_data.append(vector)
    data_length = len(raw_data[0])
    raw_data = torch.FloatTensor(raw_data)
    color_channel = 1
    width = int(math.sqrt(data_length / color_channel))
    height = int(data_length / color_channel / width)
    torch.reshape(raw_data, (raw_data.shape[0], color_channel, height, width))

    content_path = res['path']
    prepare_data(content_path, raw_data, 180, 200, 400, False)
    with open('data/2D.npy', 'rb') as f:
        result = np.load(f).tolist()

    with open('data/grid.npy', 'rb') as f:
        grid_index = np.load(f)
        grid_index = grid_index.reshape((grid_index.shape[0], -1, 2)).tolist()

    with open('data/decision_view.npy', 'rb') as f:
        grid_color = np.load(f)
        grid_color = grid_color.reshape((grid_color.shape[0], -1, 3))
        grid_color *= 255
        grid_color = grid_color.astype(int).tolist()

    with open('data/color.npy', 'rb') as f:
        standard_color = np.load(f)*255
        standard_color = standard_color.astype(int).tolist()

    with open('data/labels.pth', 'rb') as f:
        labels = torch.load(f).tolist()

    label_color_list = []
    for label in labels:
        label_color_list.append(standard_color[label])

    return make_response(jsonify({'result': result, 'grid_index': grid_index, 'grid_color': grid_color, 'label_color_list':label_color_list}), 200)

@app.route('/', methods=["POST", "GET"])
def index():
    return 'Index Page'


@app.route('/hello')
def hello_world():
    return 'Hello World!'


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":

    app.run(host="192.168.10.115")