from flask import request, Flask, jsonify, make_response
from flask_cors import CORS, cross_origin

import os
import sys
import numpy as np
import json
import torch

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
    path = os.path.normpath(res['path'])
    iteration = res['iteration']
    cache = res['cache']
    resolution = int(res['resolution'])

    p_tmp = path
    l = []
    for i in range(3):
        l.append(os.path.split(p_tmp)[1])
        p_tmp = os.path.split(p_tmp)[0]
    l.reverse()
    dir_name = "_".join(i for i in l)
    folder_path = os.path.join('data', dir_name)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    path_files = os.listdir(os.path.join(path+'Model'))
    maximum_iteration = len(path_files) - 2
    
    training_data = torch.load(os.path.join(path, "Training_data", "training_dataset_data.pth"))
    training_labels = torch.load(os.path.join(path, "Training_data", "training_dataset_label.pth"))
    testing_data = torch.load(os.path.join(path, "Testing_data", "testing_dataset_data.pth"))
    testing_labels = torch.load(os.path.join(path, "Testing_data", "testing_dataset_label.pth"))
    
    training_data_number = training_data.shape[0]
    testing_data_number = testing_data.shape[0]
    training_data_index = list(range(0, training_data_number))
    testing_data_index = list(range(training_data_number, training_data_number + testing_data_number))
    data = torch.cat((training_data, testing_data), 0)
    labels = torch.cat((training_labels, testing_labels), 0).tolist()
    
    if not cache:
        prepare_data(path, data, iteration=iteration, folder_name=folder_path, resolution=resolution, direct_call=False)

    with open(os.path.join(folder_path, 'dimension_reduction_'+str(iteration)+'.npy'), 'rb') as f:
        result = np.load(f).tolist()

    with open(os.path.join(folder_path, 'grid_'+str(iteration)+'.npy'), 'rb') as f:
        grid_index = np.load(f)
        grid_index = grid_index.reshape((-1, 2)).tolist()
    with open(os.path.join(folder_path, 'decision_view_'+str(iteration)+'.npy'), 'rb') as f:
        grid_color = np.load(f)
        grid_color = grid_color.reshape((-1, 3))
        grid_color *= 255
        grid_color = grid_color.astype(int).tolist()

    with open(os.path.join(folder_path, 'color.npy'), 'rb') as f:
        standard_color = np.load(f)*255
        standard_color = standard_color.astype(int).tolist()
    
    with open(os.path.join(folder_path, 'evaluation_'+str(iteration)+'.json'), 'r') as f:
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
    with open(os.path.join(folder_path, 'prediction_'+str(iteration)+'.npy'), 'rb') as f:
         prediction = np.load(f).tolist()
         for pred in prediction:
             prediction_list.append(classes[pred])
    
    current_training_path = os.path.join(folder_path, 'current_training_'+str(iteration)+'.json')
    if os.path.isfile(current_training_path):
         with open(current_training_path, 'r') as f:
              current_training = json.load(f)
    else:
         current_training = training_data_index

    new_selection_path = os.path.join(folder_path, 'new_selection_'+str(iteration)+'.json')
    if os.path.isfile(new_selection_path):
         with open(new_selection_path, 'r') as f:
              new_selection = json.load(f)
    else:
         new_selection = []
         
    noisy_data_path = os.path.join(folder_path, 'noisy_data_index.json')
    if os.path.isfile(noisy_data_path):
         with open(noisy_data_path, 'r') as f:
              noisy_data = json.load(f)
    else:
         noisy_data = []
    
    original_label_path = os.path.join(folder_path, 'original_label.npy')
    original_label_list = []
    if os.path.isfile(original_label_path):
         with open(original_label_path, 'rb') as f:
              original_label = np.load(f).tolist()
              for label in original_label:
                  original_label_list.append(classes[label])
    else:
         original_label_list = label_list

    with open(os.path.join(folder_path, 'acc' + str(iteration) + '.json'), 'r') as f:
        acc = json.load(f)
        acc_train = round(acc['training'], 2)
        acc_test = round(acc['testing'], 2)
        evaluation_new['acc_train'] = acc_train
        evaluation_new['acc_test'] = acc_test

    with open(os.path.join(folder_path, 'inv_acc_' + str(iteration) + '.npy'), 'rb') as f:
        inv_acc_list = np.load(f).tolist()

    uncertainty_path = os.path.join(folder_path, 'uncertainty_' + str(iteration) + '.json')
    diversity_path = os.path.join(folder_path, 'diversity_' + str(iteration) + '.json')
    tot_path = os.path.join(folder_path, 'tot_' + str(iteration) + '.json')
    is_uncertainty_diversity_tot_exist = True
    uncertainty_diversity_tot_dict = {}
    if os.path.isfile(uncertainty_path):
        with open(uncertainty_path, 'r') as f:
            uncertainty_list = json.load(f)
            uncertainty_ranking_list = [i[0] for i in sorted(enumerate(uncertainty_list), key=lambda x:x[1])]
        with open(diversity_path, 'r') as f:
            diversity_list = json.load(f)
            diversity_ranking_list = [i[0] for i in sorted(enumerate(diversity_list), key=lambda x: x[1])]
        with open(tot_path, 'r') as f:
            tot_list = json.load(f)
            tot_ranking_list = [i[0] for i in sorted(enumerate(tot_list), key=lambda x: x[1])]
        uncertainty_diversity_tot_dict['uncertainty'] = uncertainty_list
        uncertainty_diversity_tot_dict['diversity'] = diversity_list
        uncertainty_diversity_tot_dict['tot'] = tot_list
        uncertainty_diversity_tot_dict['uncertainty_ranking'] = uncertainty_ranking_list
        uncertainty_diversity_tot_dict['diversity_ranking'] = diversity_ranking_list
        uncertainty_diversity_tot_dict['tot_ranking'] = tot_ranking_list
    else:
        is_uncertainty_diversity_tot_exist = False
    uncertainty_diversity_tot_dict['is_exist'] = is_uncertainty_diversity_tot_exist

    return make_response(jsonify({'result': result, 'grid_index': grid_index, 'grid_color': grid_color,
                                  'label_color_list':label_color_list, 'label_list':label_list,
                                  'maximum_iteration':maximum_iteration, 'training_data':current_training,
                                  'testing_data':testing_data_index, 'evaluation':evaluation_new,
                                  'prediction_list':prediction_list, 'new_selection':new_selection,
                                  'noisy_data':noisy_data, 'original_label_list':original_label_list,
                                  'inv_acc_list':inv_acc_list,
                                  'uncertainty_diversity_tot':uncertainty_diversity_tot_dict}), 200)

@app.route('/', methods=["POST", "GET"])
def index():
    return 'Index Page'


@app.route('/hello')
def hello_world():
    return 'Hello World!'


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    with open('../tensorboard/tensorboard/plugins/projector/vz_projector/standalone_projector_config.json', 'r') as f:
        ip_adress = json.load(f)["serverIp"]
    app.run(host=ip_adress)
