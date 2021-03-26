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
    
    training_data = torch.load(data_path + "/training_dataset_data.pth")
    training_labels = torch.load(data_path + "/training_dataset_label.pth")
    testing_data = torch.load(data_path + "/testing_dataset_data.pth")
    testing_labels = torch.load(data_path + "/testing_dataset_label.pth")
    
    training_data_number = training_data.shape[0]
    testing_data_number = testing_data.shape[0]
    training_data_index = list(range(0, training_data_number))
    testing_data_index = list(range(training_data_number, training_data_number + testing_data_number))
    data = torch.cat((training_data, testing_data), 0)
    labels = torch.cat((training_labels, testing_labels), 0).tolist()
    
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
    
    with open(folder_path+'/evaluation_'+str(iteration)+'.json', 'r') as f:
        evaluation = json.load(f)
        evaluation_new = evaluation
        for item in evaluation:
          value = evaluation[item]
          value = round(value, 2)
          evaluation_new[item] = value    

    label_color_list = []
    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    label_list = []
    for label in labels:
        label_color_list.append(standard_color[label])
        label_list.append(classes[label])
    
    prediction_list = []
    with open(folder_path+'/prediction_'+str(iteration)+'.npy', 'rb') as f:
         prediction = np.load(f).tolist()
         for pred in prediction:
             prediction_list.append(classes[pred])
    '''
    with open(folder_path+'/current_training_'+str(iteration)+'.json', 'r') as f:
         current_training = json.load(f)
    
    with open(folder_path+'/new_selection_'+str(iteration)+'.json', 'r') as f:
         new_selection = json.load(f)
    '''  
    with open(folder_path+'/noisy_data_index.json','r') as f:
         noisy_data = json.load(f)
    
    original_label_list = []
    with open(folder_path+'/original_label.npy', 'rb') as f:
         original_label = np.load(f).tolist()
         for label in original_label:
             original_label_list.append(classes[label])
             
    return make_response(jsonify({'result': result, 'grid_index': grid_index, 'grid_color': grid_color, 'label_color_list':label_color_list, 'label_list':label_list, 'maximum_iteration':maximum_iteration, 'training_data':training_data_index, 'testing_data':testing_data_index, 'evaluation':evaluation_new, 'prediction_list':prediction_list, 'new_selection':[], 'noisy_data':noisy_data, 'original_label_list':original_label_list}), 200)

@app.route('/', methods=["POST", "GET"])
def index():
    return 'Index Page'


@app.route('/hello')
def hello_world():
    return 'Hello World!'


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":

    app.run(host="192.168.254.128")
