from flask import request, Response, Flask, jsonify, make_response, send_file
from flask_cors import CORS, cross_origin

import os
import sys
import numpy as np
import time
import json
import torch
import tensorflow as tf
from PIL import Image
sys.path.append("..")
from deepvisualinsight.MMS import MMS

from sqlalchemy import create_engine, text
import pymysql
import pandas as pd
from antlr4 import *
from MyGrammar.MyGrammarLexer import MyGrammarLexer
from MyGrammar.MyGrammarListener import MyGrammarListener
from MyGrammar.MyGrammarParser import MyGrammarParser

#from prepare_data import prepare_data

# flask for API server
app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

class MyGrammarPrintListener(MyGrammarListener):
    def __init__(self, epochs=[1], selected_epoch=None):
        self.epochs = epochs
        if selected_epoch:
            self.selected_epoch = selected_epoch
        else:
            self.selected_epoch = self.epochs[0]
        self.result = ""
        self.pred_sample_needed = False
        self.epoch_sample_needed = False
        self.deltab_sample_needed = False
        self.stack = []
        self.array = []

    def enterMultiplecond2(self, ctx):
        if ctx.CONDOP().getText() == "&":
            self.result += " AND "
        elif ctx.CONDOP().getText() == "|":
            self.result += " OR "

    def enterParencond1(self, ctx):
        self.result += "("

    def exitParencond1(self, ctx):
        self.result += ")"

    def exitCond2(self, ctx):
        result = ""
        right = self.stack.pop()
        if self.stack:
            left  = self.stack.pop()
            result =  left + str(ctx.OP()) + right
        elif self.array:
            left = right
            lenArray = len(self.array)
            for _ in range(lenArray):
                i = self.array.pop(0)
                result += i +","
            result = left + " IN (" + result[:-1] + ")"       
        self.result += result

    def exitArray(self, ctx):
        for i in ctx.INT():
            self.array.append(i.getText())

    def exitParameter(self, ctx):
        if ctx.STRING():
            self.stack.append(self.checkString(ctx.STRING().getText()))

    def exitPositive(self, ctx):
        if ctx.INT():
            self.stack.append(ctx.INT().getText())

    def exitNegative(self, ctx):
    	# Return the number of indexes for an array based on the negative integer value
        if ctx.INT():
            value = int("-"+ctx.INT().getText())
            for i in self.epochs[value:]:
                self.array.append(str(i))

    def exitExpr(self, ctx):
    	# MYSQL statement is built here
        if "search for samples" in ctx.ACTION().getText():
            action = "SELECT Sample.idx FROM Sample "
        if self.result:
            self.result = "WHERE " + self.result
        if self.pred_sample_needed or self.epoch_sample_needed or self.deltab_sample_needed:
            self.result = action + "INNER JOIN PredSample ON Sample.idx =  PredSample.idx " + self.result
        else:
            self.result = action + self.result
        if (self.pred_sample_needed or self.deltab_sample_needed) and not self.epoch_sample_needed:
            self.result += " AND PredSample.epoch=" + str(self.selected_epoch)
        elif self.epoch_sample_needed:
            self.result += " GROUP BY Sample.idx"
        self.result += ";"

    def checkString(self, string):
    	# check the strings and categorize them for MYSQL statement
        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        if "pred" in string:
            result = string.split(".")
            self.pred_sample_needed = True
            return "PredSample."+result[1]
        elif "epoch" in string:
            result = string.split(".")
            self.epoch_sample_needed = True
            return "PredSample."+result[1]
        elif "deltab" in string:
            result = string.split(".")
            self.deltab_sample_needed = True
            return "PredSample."+result[1]
        elif "sample" in string:
            result = string.split(".")
            return "Sample."+result[1]
        elif string in classes:
            return str(classes.index(string))
        elif string in ["test","train"]:
            return "'"+string+"'"
        elif string == "false":
            return "0"
        elif string == "true":
            return "1"
        else:
            return string + " "

@app.route('/animation', methods=["POST"])
@cross_origin()
def animation():
    res = request.get_json()
    path = os.path.normpath(res['path'])
    iteration = res['iteration']
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
    
    path_files = os.listdir(os.path.join(path, 'Model'))
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
        label_color_list.append(standard_color[int(label)])
        label_list.append(classes[int(label)])
    
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


def record(string):
    with open("record.txt", "a") as file_object:
        file_object.write(string+"\n")

@app.route('/load', methods=["POST", "GET"])
@cross_origin()
def load():
    t1 = time.time()

    res = request.get_json()
    content_path = os.path.normpath(res['path'])

    sys.path.append(content_path)

    try:
        from Model.model import ResNet18
        net = ResNet18()
    except:
        from Model.model import resnet18
        net = resnet18()

    # Retrieving hyperparameters from json file to be passed as  parameters for MMS model
    with open(content_path+"/config.json") as file:
        data = json.load(file)
        for key in data:
            if key=="dataset":
                dataset = data[key]
            elif key=="epoch_start":
                start_epoch = int(data[key])
            elif key=="epoch_end":
                end_epoch = int(data[key])
            elif key=="epoch_period":
                period = int(data[key])
            elif key=="embedding_dim":
                embedding_dim = int(data[key])
            elif key=="num_classes":
                num_classes = int(data[key])
            elif key=="classes":
                classes = range(num_classes)
            elif key=="temperature":
                temperature = float(data[key])
            elif key=="attention":
                if int(data[key]) == 0:
                    attention = False
                else:
                    attention = True
            elif key=="cmap":
                cmap = data[key]
            elif key=="resolution":
                resolution = int(data[key])
            elif key=="temporal":
                if int(data[key]) == 0:
                    temporal = False
                else:
                    temporal = True
            elif key=="transfer_learning":
                transfer_learning = int(data[key])
            elif key=="step3":
                step3 = int(data[key])
            elif key=="split":
                split = int(data[key])
            elif key=="advance_border_gen":
                if int(data[key]) == 0:
                    advance_border_gen = False
                else:
                    advance_border_gen = True
            elif key=="alpha":
                alpha = float(data[key])
            elif key=="withoutB":
                if int(data[key]) == 0:
                    withoutB = False
                else:
                    withoutB = True
            elif key=="attack_device":
                attack_device = data[key]


    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    mms = MMS(content_path, net, start_epoch, end_epoch, period, embedding_dim, num_classes, classes, temperature=temperature, cmap=cmap, resolution=resolution, verbose=1,
              temporal=temporal, split=split, alpha=alpha, advance_border_gen=advance_border_gen, withoutB=withoutB, attack_device="cpu")

    sql_engine       = create_engine('mysql+pymysql://xg:password@localhost/dviDB', pool_recycle=3600)
    db_connection    = sql_engine.connect()

    # Search the following tables in MYSQL database and drop them if they exist
    sql_engine.execute(text('DROP TABLE IF EXISTS SubjectModel;'))
    sql_engine.execute(text('DROP TABLE IF EXISTS VisModel;'))
    sql_engine.execute(text('DROP TABLE IF EXISTS Sample;'))
    sql_engine.execute(text('DROP TABLE IF EXISTS NoisySample;'))
    sql_engine.execute(text('DROP TABLE IF EXISTS AlSample;'))
    sql_engine.execute(text('DROP TABLE IF EXISTS PredSample;'))

    # Create the SubjectModel table in MYSQL database and insert the data
    table_subject_model = "SubjectModel"
    data_subject_model = mms.subject_model_table()
    data_subject_model.to_sql(table_subject_model, db_connection, if_exists='fail');

    # Create the VisModel table in MYSQL database and insert the data
    table_vis_model = "VisModel"
    data_vis_model = mms.vis_model_table()
    data_vis_model.to_sql(table_vis_model, db_connection, if_exists='fail');

    # Create the Sample table in MYSQL database and insert the data
    table_sample = "Sample"
    data_sample = mms.sample_table()
    data_sample.to_sql(table_sample, db_connection, if_exists='fail');

    # For nosiy or active learning data, currently not tested yet
    if "noisy" in content_path:     
        table_noisy_sample = "NoisySample"
        data_noisy_sample = mms.sample_table_noisy()
        data_noisy_sample.to_sql(table_noisy_sample, db_connection, if_exists='fail');
    elif "active" in content_path:
        table_al_sample = "AlSample"
        data_al_sample = mms.sample_table_AL()
        data_al_sample.to_sql(table_al_sample, db_connection, if_exists='fail');
    
    # Ablation starts here
    # Store prediction, deltaboundary true/false for all samples in all epochs in PredSample table
    all_prediction_list = []
    all_deltab_list = []
    all_epochs_list = []
    all_idx_list = []
    for iteration in range(start_epoch, end_epoch+1, period): 
        print("iteration", iteration)
        train_data = mms.get_data_pool_repr(iteration)
        test_data = mms.get_epoch_test_repr_data(iteration)
        all_data = np.concatenate((train_data, test_data), axis=0)

        prediction = mms.get_pred(iteration, all_data).argmax(-1)
        deltab = mms.is_deltaB(iteration, all_data)
        count = 0
        for idx,_ in enumerate(prediction):
            all_prediction_list.append(prediction[idx])
            all_deltab_list.append(deltab[idx])
            all_epochs_list.append(iteration)
            all_idx_list.append(count)
            count += 1

    data_pred_sample = pd.DataFrame(list(zip(all_idx_list, all_epochs_list, all_prediction_list, all_deltab_list)),
               columns =['idx', 'epoch', 'pred', 'deltab'])
    table_pred_sample = "PredSample"
    data_pred_sample.to_sql(table_pred_sample, db_connection, if_exists='fail');
    # Ablation ends here

    db_connection.close()

    t2 = time.time()
    print("time taken", t2-t1)
    save_to_file = "{:.2f}".format(t2-t1)+"|"+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))+"|"+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t2))+"|loadPredDeltaBKNN|"+content_path
    record(save_to_file)
    return None


@app.route('/updateProjection', methods=["POST", "GET"])
@cross_origin()
def update_projection():
    res = request.get_json()
    content_path = os.path.normpath(res['path'])
    resolution = int(res['resolution'])
    predicates = res["predicates"]
    sys.path.append(content_path)

    try:
        from Model.model import ResNet18
        net = ResNet18()
    except:
        from Model.model import resnet18
        net = resnet18()

    # Retrieving hyperparameters from json file to be passed as  parameters for MMS model
    with open(content_path+"/config.json") as file:
        data = json.load(file)
        for key in data:
            if key=="dataset":
                dataset = data[key]
            elif key=="epoch_start":
                start_epoch = int(data[key])
            elif key=="epoch_end":
                end_epoch = int(data[key])
            elif key=="epoch_period":
                period = int(data[key])
            elif key=="embedding_dim":
                embedding_dim = int(data[key])
            elif key=="num_classes":
                num_classes = int(data[key])
            elif key=="classes":
                classes = range(num_classes)
            elif key=="temperature":
                temperature = float(data[key])
            elif key=="attention":
                if int(data[key]) == 0:
                    attention = False
                else:
                    attention = True
            elif key=="cmap":
                cmap = data[key]
            elif key=="resolution":
                resolution = int(data[key])
            elif key=="temporal":
                if int(data[key]) == 0:
                    temporal = False
                else:
                    temporal = True
            elif key=="transfer_learning":
                transfer_learning = int(data[key])
            elif key=="step3":
                step3 = int(data[key])
            elif key=="split":
                split = int(data[key])
            elif key=="advance_border_gen":
                if int(data[key]) == 0:
                    advance_border_gen = False
                else:
                    advance_border_gen = True
            elif key=="alpha":
                alpha = float(data[key])
            elif key=="withoutB":
                if int(data[key]) == 0:
                    withoutB = False
                else:
                    withoutB = True
            elif key=="attack_device":
                attack_device = data[key]

    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    mms = MMS(content_path, net, start_epoch, end_epoch, period, embedding_dim, num_classes, classes, temperature=temperature, cmap=cmap, resolution=resolution, verbose=1, attention=attention, 
              temporal=False, split=split, alpha=alpha, advance_border_gen=advance_border_gen, withoutB=withoutB, attack_device="cpu")
    iteration = int(res['iteration'])*period


    train_data = mms.get_data_pool_repr(iteration)
    # train_data = mms.get_epoch_train_repr_data(iteration)
    test_data = mms.get_epoch_test_repr_data(iteration)
    all_data = np.concatenate((train_data, test_data), axis=0)

    embedding_2d = mms.batch_project(all_data, iteration).tolist()
    train_labels = mms.training_labels.cpu().numpy()
    test_labels = mms.testing_labels.cpu().numpy()
    labels = np.concatenate((train_labels, test_labels),axis=0).tolist()

    training_data_number = train_data.shape[0]
    testing_data_number = test_data.shape[0]
    testing_data_index = list(range(training_data_number, training_data_number + testing_data_number))

    grid, decision_view = mms.get_epoch_decision_view(iteration, resolution)

    grid = grid.reshape((-1, 2)).tolist()
    decision_view = decision_view * 255
    decision_view = decision_view.reshape((-1, 3)).astype(int).tolist()

    color = mms.get_standard_classes_color() * 255
    color = color.astype(int).tolist()

    evaluation = mms.get_eval(iteration)

    label_color_list = []
    label_list = []
    for label in labels:
        label_color_list.append(color[int(label)])
        label_list.append(classes[int(label)])

    prediction_list = []
    prediction = mms.get_pred(iteration, all_data).argmax(-1)
    # classes_map = dict()
    # for i in range(10):
    #     classes_map[i] = classes[i]
    # prediction_list = np.vectorize(classes_map.get)(prediction).tolist()
    for pred in prediction:
        prediction_list.append(classes[pred])

    max_iter = 0
    path_files = os.listdir(mms.model_path)
    for file in path_files:
        if "Epoch" in file:
            max_iter += 1
        if "Epoch_0" in file:
            max_iter -= 1

    _, conf_diff = mms.batch_inv_preserve(iteration, all_data)
    current_index = mms.get_epoch_index(iteration)

    new_index = mms.get_new_index(iteration)

    noisy_data = mms.noisy_data_index()

    original_labels = mms.get_original_labels()
    original_label_list = []
    for label in original_labels:
        original_label_list.append(classes[label])

    uncertainty_diversity_tot_dict = {}
    uncertainty_diversity_tot_dict['uncertainty'] = mms.get_uncertainty_score(iteration)
    uncertainty_diversity_tot_dict['diversity'] = mms.get_diversity_score(iteration)
    uncertainty_diversity_tot_dict['tot'] = mms.get_total_score(iteration)

    uncertainty_ranking_list = [i[0] for i in sorted(enumerate(uncertainty_diversity_tot_dict['uncertainty']), key=lambda x: x[1])]
    diversity_ranking_list = [i[0] for i in sorted(enumerate(uncertainty_diversity_tot_dict['diversity']), key=lambda x: x[1])]
    tot_ranking_list = [i[0] for i in sorted(enumerate(uncertainty_diversity_tot_dict['tot']), key=lambda x: x[1])]
    uncertainty_diversity_tot_dict['uncertainty_ranking'] = uncertainty_ranking_list
    uncertainty_diversity_tot_dict['diversity_ranking'] = diversity_ranking_list
    uncertainty_diversity_tot_dict['tot_ranking'] = tot_ranking_list

    selected_points = np.arange(mms.get_dataset_length())
    for key in predicates.keys():
        if key == "new_selection":
            tmp = np.array(mms.get_new_index(int(predicates[key])))
        elif key == "label":
            tmp = np.array(mms.filter_label(predicates[key]))
        elif key == "type":
            tmp = np.array(mms.filter_type(predicates[key], int(iteration)))
        else:
            tmp = np.arange(mms.get_dataset_length())
        selected_points = np.intersect1d(selected_points, tmp)

    sys.path.remove(content_path)


    return make_response(jsonify({'result': embedding_2d, 'grid_index': grid, 'grid_color': decision_view,
                                  'label_color_list': label_color_list, 'label_list': label_list,
                                  'maximum_iteration': max_iter, 'training_data': current_index,
                                  'testing_data': testing_data_index, 'evaluation': evaluation,
                                  'prediction_list': prediction_list, 'new_selection': new_index,
                                  'noisy_data': noisy_data, 'original_label_list': original_label_list,
                                  'inv_acc_list': conf_diff.tolist(),
                                  'uncertainty_diversity_tot': uncertainty_diversity_tot_dict,
                                  "selectedPoints":selected_points.tolist()}), 200)

@app.route('/query', methods=["POST"])
@cross_origin()
def filter():
    res = request.get_json()
    string = res["predicates"]["label"]
    content_path = os.path.normpath(res['content_path'])

    data =  InputStream(string)
    # lexer
    lexer = MyGrammarLexer(data)
    stream = CommonTokenStream(lexer)
    # parser
    parser = MyGrammarParser(stream)
    tree = parser.expr()
    # Currently this is hardcoded for CIFAR10, changes need to be made in future
    # Error will appear based on some of the queries sent
    model_epochs = [40, 80, 120, 160, 200]
    # evaluator
    listener = MyGrammarPrintListener(model_epochs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    statement = listener.result

    sql_engine       = create_engine('mysql+pymysql://xg:password@localhost/dviDB', pool_recycle=3600)
    db_connection    = sql_engine.connect()
    frame           = pd.read_sql(statement, db_connection);
    pd.set_option('display.expand_frame_repr', False)
    db_connection.close()
    result = []
    for _, row in frame.iterrows():
        for col in frame.columns:
            result.append(int(row[col]))
    return make_response(jsonify({"selectedPoints":result}), 200)


@app.route('/saveDVIselections', methods=["POST"])
@cross_origin()
def save_DVI_selections():
    data = request.get_json()
    indices = data["newIndices"]

    content_path = os.path.normpath(data['content_path'])
    iteration = data["iteration"]
    sys.path.append(content_path)
    try:
        from Model.model import ResNet18
        net = ResNet18()
    except:
        from Model.model import resnet18
        net = resnet18()

    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    mms = MMS(content_path, net, 1, 20, 1, 512, 10, classes, cmap="tab10", neurons=256, verbose=1,
              temporal=False, split=-1, advance_border_gen=True, attack_device="cpu")
    mms.save_DVI_selection(iteration, indices)

    sys.path.remove(content_path)

    return make_response(jsonify({"message":"Save DVI selection succefully!"}), 200)

@app.route('/sprite', methods=["POST","GET"])
@cross_origin()
def sprite_image():
    path= request.args.get("path")
    sprite = tf.io.gfile.GFile(path, "rb")
    encoded_image_string = sprite.read()
    sprite.close()
    image_type = "image/png"
    return Response(encoded_image_string, status=200, mimetype=image_type)

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    with open('../tensorboard/tensorboard/plugins/projector/vz_projector/standalone_projector_config.json', 'r') as f:
        config = json.load(f)
        ip_adress = config["DVIServerIP"]
        port = config["DVIServerPort"]
    app.run(host=ip_adress, port=int(port))
