import os

# DeepView related
# for i in range(1, 11, 1):
# #     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\cifar10 --epoch_start 40 --epoch_end 200 --epoch_period 40 --embedding_dim 512 --dataset CIFAR10 --cuda True -a 0.6 -t 0.3".format(i))
#     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\cifar10 --epoch_start 40 --epoch_end 200 --epoch_period 40 --resolution 100 --embedding_dim 512 --cuda True -a 0.6 -t 0.3 --num_class 10 --split -1 --temporal 3 --preprocess 0".format(i))
# for i in range(1, 11, 1):
# #     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\mnist --epoch_start 4 --epoch_end 20 --epoch_period 4 --embedding_dim 512 --dataset MNIST --cuda True -a 0.5 -t 0.02".format(i))
#     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\mnist --epoch_start 4 --epoch_end 20 --epoch_period 4 --resolution 100 --embedding_dim 512 --cuda True -a 0.5 -t 0.02 --num_class 10 --split -1 --temporal 3 --preprocess 0".format(i))
# for i in range(1, 11, 1):
# #     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\fmnist --epoch_start 10 --epoch_end 50 --epoch_period 10 --embedding_dim 512 --dataset FashionMNIST --cuda True -a 0.5 -t 0.05".format(i))
#     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\fmnist --epoch_start 10 --epoch_end 50 --epoch_period 10 --resolution 100 --embedding_dim 512 --cuda True -a 0.5 -t 0.1 --num_class 10 --split -1 --temporal 3 --preprocess 0".format(i))
#
# for i in range(1, 11, 1):
# #     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\resnet50 --epoch_start 40 --epoch_end 200 --epoch_period 40 --embedding_dim 2048 --dataset CIFAR10 --cuda True -a 0.6 -t 0.1".format(i))
#     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\resnet50 --epoch_start 40 --epoch_end 200 --epoch_period 40 --resolution 100 --embedding_dim 2048 --cuda True -a 0.6 -t 0.15 --num_class 10 --split -1 --temporal 3 --preprocess 0".format(i))

# DVI exp
# cifar10, different strategies:
# non transfer, transfer, step2
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_cifar10 --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 512 --cuda True -a 0.6 -t 0.3 --num_class 10 --split -1 --temporal 1 --preprocess 1")
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_cifar10 --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 512 --cuda True -a 0.6 -t 0.3 --num_class 10 --split -1 --temporal 2 --preprocess 0")
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_cifar10 --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 512 --cuda True -a 0.6 -t 0.3 --num_class 10 --split -1 --temporal 3 --preprocess 0")
#
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_mnist --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 512 --cuda True -a 0.5 -t 0.02 --num_class 10 --split -1 --temporal 1 --preprocess 1")
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_mnist --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 512 --cuda True -a 0.5 -t 0.02 --num_class 10 --split -1 --temporal 2 --preprocess 0")
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_mnist --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 512 --cuda True -a 0.5 -t 0.02 --num_class 10 --split -1 --temporal 3 --preprocess 0")
#
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_fmnist --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 512 --cuda True -a 0.5 -t 0.1 --num_class 10 --split -1 --temporal 1 --preprocess 1")
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_fmnist --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 512 --cuda True -a 0.5 -t 0.1 --num_class 10 --split -1 --temporal 2 --preprocess 0")
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet18_fmnist --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 512 --cuda True -a 0.5 -t 0.1 --num_class 10 --split -1 --temporal 3 --preprocess 0")
#
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet50_cifar10 --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 2048 --cuda True -a 0.6 -t 0.15 --num_class 10 --split -1 --temporal 1 --preprocess 1")
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet50_cifar10 --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 2048 --cuda True -a 0.6 -t 0.15 --num_class 10 --split -1 --temporal 2 --preprocess 0")
# os.system("python run.py --content_path E:\\DVI_exp_data\\TemporalExp\\resnet50_cifar10 --epoch_start 1 --epoch_end 7 --epoch_period 1 --resolution 100 --embedding_dim 2048 --cuda True -a 0.6 -t 0.15 --num_class 10 --split -1 --temporal 3 --preprocess 0")

# baseline
os.system("python run.py --content_path E:\\DVI_exp_data\\resnet18_cifar10 --epoch_start 40 --epoch_end 200 --epoch_period 40 --resolution 100 --embedding_dim 512 --cuda True -a 0.6 -t 0.3 --num_class 10 --split -1 --temporal 1 --preprocess 1 --parametricUmap 1")
os.system("python run.py --content_path E:\\DVI_exp_data\\resnet18_mnist --epoch_start 4 --epoch_end 20 --epoch_period 4 --resolution 100 --embedding_dim 512 --cuda True -a 0.5 -t 0.02 --num_class 10 --split -1 --temporal 1 --preprocess 1 --parametricUmap 1")
os.system("python run.py --content_path E:\\DVI_exp_data\\resnet18_fmnist --epoch_start 10 --epoch_end 50 --epoch_period 10 --resolution 100 --embedding_dim 512 --cuda True -a 0.5 -t 0.1 --num_class 10 --split -1 --temporal 1 --preprocess 1 --parametricUmap 1")