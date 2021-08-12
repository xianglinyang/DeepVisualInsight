import os

# for i in range(1, 11, 1):
#     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\cifar10 --epoch_start 40 --epoch_end 200 --epoch_period 40 --embedding_dim 512 --dataset CIFAR10 --cuda True -a 0.6 -t 0.3".format(i))
#
# for i in range(1, 11, 1):
#     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\mnist --epoch_start 4 --epoch_end 20 --epoch_period 4 --embedding_dim 512 --dataset MNIST --cuda True -a 0.5 -t 0.02".format(i))

# for i in range(1, 11, 1):
#     os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\fmnist --epoch_start 10 --epoch_end 50 --epoch_period 10 --embedding_dim 512 --dataset FashionMNIST --cuda True -a 0.5 -t 0.05".format(i))

for i in range(1, 11, 1):
    os.system("python run.py --content_path E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}\\resnet50 --epoch_start 40 --epoch_end 200 --epoch_period 40 --embedding_dim 2048 --dataset CIFAR10 --cuda True -a 0.6 -t 0.1".format(i))