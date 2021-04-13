from deepvisualinsight.MMS import MMS
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

def prepare_data(content_path, data, iteration, resolution, folder_name, direct_call=True):
    sys.path.append(content_path)
    net = None
    try:
        from Model.model import resnet18
        net = resnet18()
    except Exception as e:
        from Model.model import ResNet18
        net = ResNet18()

    prefix = folder_name + '/'
    if direct_call:
        prefix = 'server/'+folder_name + '/'

    #net = ResNet18()
  


    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    mms = MMS(content_path, net, 1, 11, 1, 512, 10, classes, cmap="tab10", resolution=resolution, neurons=256,
              verbose=1)
    # mms.data_preprocessing()
    # mms.prepare_visualization_for_all()
    #new_index = mms.get_new_index(iteration)
    #current_index = mms.get_epoch_index(iteration)
    #print(len(new_index), len(current_index))
    '''
    with open(prefix+'new_selection_'+str(iteration)+'.json', 'w') as f:
        json.dump(new_index, f)
    '''
    '''
    with open(prefix+'current_training_'+str(iteration)+'.json', 'w') as f:
        json.dump(current_index, f)
    '''
    noisy_data = mms.noisy_data_index()
    with open(prefix+'noisy_data_index.json','w') as f:
        json.dump(noisy_data, f)
    original_label = mms.get_original_labels()
    with open(prefix+'original_label.npy', 'wb') as f:
        np.save(f, original_label)
    '''
    evaluation = {}
    evaluation['nn_train_10'] = mms.proj_nn_perseverance_knn_train(iteration, 10)
    evaluation['nn_train_15'] = mms.proj_nn_perseverance_knn_train(iteration, 15)
    evaluation['nn_train_30'] = mms.proj_nn_perseverance_knn_train(iteration, 30)
    evaluation['nn_test_10'] = mms.proj_nn_perseverance_knn_test(iteration, 10)
    evaluation['nn_test_15'] = mms.proj_nn_perseverance_knn_test(iteration, 15)
    evaluation['nn_test_30'] = mms.proj_nn_perseverance_knn_test(iteration, 30)
    evaluation['bound_train_10'] = mms.proj_boundary_perseverance_knn_train(iteration, 10)
    evaluation['bound_train_15'] = mms.proj_boundary_perseverance_knn_train(iteration, 15)
    evaluation['bound_train_30'] = mms.proj_boundary_perseverance_knn_train(iteration, 30)
    evaluation['bound_test_10'] = mms.proj_boundary_perseverance_knn_test(iteration, 10)
    evaluation['bound_test_15'] = mms.proj_boundary_perseverance_knn_test(iteration, 15)
    evaluation['bound_test_30'] = mms.proj_boundary_perseverance_knn_test(iteration, 30)
    print("finish half of it")
    evaluation['inv_nn_train_10'] = mms.inv_nn_preserve_train(iteration, 10)
    evaluation['inv_nn_train_15'] = mms.inv_nn_preserve_train(iteration, 15)
    evaluation['inv_nn_train_30'] = mms.inv_nn_preserve_train(iteration, 30)
    evaluation['inv_nn_test_10'] = mms.inv_nn_preserve_test(iteration, 10)
    evaluation['inv_nn_test_15'] = mms.inv_nn_preserve_test(iteration, 15)
    evaluation['inv_nn_test_30'] = mms.inv_nn_preserve_test(iteration, 30)
    evaluation['inv_acc_train'] = mms.inv_accu_train(iteration)
    evaluation['inv_acc_test'] = mms.inv_accu_test(iteration)
    evaluation['inv_conf_train'] = mms.inv_conf_diff_train(iteration)
    evaluation['inv_conf_test'] = mms.inv_conf_diff_test(iteration)
    with open(prefix+'evaluation_'+str(iteration)+'.json', 'w') as f:
        json.dump(evaluation, f)
        
    gap_layer_data = mms.get_representation_data(iteration, data)
    prediction = mms.get_pred(iteration, gap_layer_data).argmax(-1)
    with open(prefix+'prediction_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, prediction)
        
    dimension_reduction_result = mms.batch_get_embedding(data, iteration)

    with open(prefix+'dimension_reduction_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, dimension_reduction_result)

    grid, decision_view = mms.get_epoch_decision_view(iteration, resolution)


    with open(prefix+'grid_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, grid)

    with open(prefix+'decision_view_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, decision_view)

    color = mms.get_standard_classes_color()

    with open(prefix+'color.npy', 'wb') as f:
        np.save(f, color)
    '''   
if __name__ == "__main__":
    content_path = "/home/selab/Enviroment/data/noisy"
    training_data = torch.load("/home/selab/Enviroment/data/noisy/data/training_dataset_data.pth")
    testing_data = torch.load("/home/selab/Enviroment/data/noisy/data/testing_dataset_data.pth")
    data = torch.cat((training_data, testing_data), 0)
    print("start")
    for i in range(1, 12):
      print("prepare for iteration: " + str(i))
      prepare_data(content_path, data, iteration=i, folder_name="data/noisy", resolution=200)
