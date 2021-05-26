from deepvisualinsight.MMS import MMS
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import argparse

def prepare_data(content_path, data, iteration, resolution, folder_name, direct_call=True, method='active_learning'):
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
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    mms = MMS(content_path, net, 1, 11, 1, 512, 10, classes, cmap="tab10", resolution=resolution, neurons=256,
              verbose=1)

    # active learning
    if method == 'active_learning':
        new_index = mms.get_new_index(iteration)
        current_index = mms.get_epoch_index(iteration)
        with open(prefix + 'new_selection_' + str(iteration) + '.json', 'w') as f:
            json.dump(new_index, f)
        with open(prefix + 'current_training_' + str(iteration) + '.json', 'w') as f:
            json.dump(current_index, f)

        # uncertainty score
        uncertainty = mms.get_uncertainty_score(iteration)
        with open(prefix+'uncertainty_'+str(iteration)+'.json', 'w') as f:
            json.dump(uncertainty, f)

        # diversity score
        diversity = mms.get_diversity_score(iteration)
        with open(prefix + "diversity_"+str(iteration)+".json", 'w') as f:
            json.dump(diversity, f)

        # total score
        tot = mms.get_total_score(iteration)
        with open(prefix+"tot_"+str(iteration)+".json",'w') as f:
            json.dump(tot, f)

    # accu
    training_acc = mms.training_accu(iteration)
    testing_acc = mms.testing_accu(iteration)
    with open(prefix+'acc'+str(iteration)+'.json', 'w') as f:
        json.dump({'training': training_acc, 'testing':testing_acc}, f)

    # training with noisy data
    if method == 'noisy':
        noisy_data = mms.noisy_data_index()
        with open(prefix + 'noisy_data_index.json', 'w') as f:
            json.dump(noisy_data, f)
        original_label = mms.get_original_labels()
        with open(prefix + 'original_label.npy', 'wb') as f:
            np.save(f, original_label)

    # evaluation information
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
    print("finish proj eval")

    evaluation['inv_acc_train'] = mms.inv_accu_train(iteration)
    evaluation['inv_acc_test'] = mms.inv_accu_test(iteration)
    evaluation['inv_conf_train'] = mms.inv_conf_diff_train(iteration)
    evaluation['inv_conf_test'] = mms.inv_conf_diff_test(iteration)
    print("finish inv eval")
    with open(prefix+'evaluation_'+str(iteration)+'.json', 'w') as f:
        json.dump(evaluation, f)

    # prediction result && inverse accuracy
    gap_layer_data = mms.get_representation_data(iteration, data)
    _, conf_diff = mms.batch_inv_preserve(iteration, gap_layer_data)
    with open(prefix + 'inv_acc_' + str(iteration) + '.npy', 'wb') as f:
        np.save(f, conf_diff)

    prediction = mms.get_pred(iteration, gap_layer_data).argmax(-1)
    with open(prefix+'prediction_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, prediction)


    # dimensionality reduction result
    dimension_reduction_result = mms.batch_embedding(data, iteration)
    with open(prefix+'dimension_reduction_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, dimension_reduction_result)

    # grid point inverse mapping
    grid, decision_view = mms.get_epoch_decision_view(iteration, resolution)
    with open(prefix+'grid_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, grid)

    with open(prefix+'decision_view_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, decision_view)

    # standard color
    color = mms.get_standard_classes_color()
    with open(prefix+'color.npy', 'wb') as f:
        np.save(f, color)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Information needed for prepare data')
    parser.add_argument('--dir_path', type=str, default='/models/data/', help='path to the dataset')
    parser.add_argument('--dir_name', type=str, default='entropy', help='dataset name')
    parser.add_argument('--iteration_number', type=int, default=6, help='number of iterations')
    parser.add_argument('--resolution', type=int, default=200, help='resolution for background')
    parser.add_argument('--method', type=str, default='active_learning', help='normal/active_learning/noisy')
    args = parser.parse_args()
    print(args)

    content_path = args.dir_path + args.dir_name
    training_data = torch.load(args.dir_path + args.dir_name + "/data/training_dataset_data.pth")
    testing_data = torch.load(args.dir_path + args.dir_name + "/data/testing_dataset_data.pth")
    data = torch.cat((training_data, testing_data), 0)
    print("start")
    for i in range(1, args.iteration_number + 1):
        print("prepare for iteration: " + str(i))
        prepare_data(content_path, data, iteration=i, folder_name="data/"+args.dir_name, resolution=args.resolution,
                     method=args.method)
