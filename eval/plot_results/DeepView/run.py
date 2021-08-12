import os
os.system("python nn.py --dataset cifar10 -s 40 -e 240 -p 40")

# for type in ["nn", "boundary", "inv_accu", "inv_conf"]:
    # os.system("python {}.py --dataset mnist -s 4 -e 24 -p 4".format(type))
    # os.system("python {}.py --dataset fmnist -s 10 -e 60 -p 10".format(type))
    # os.system("python {}.py --dataset cifar10 -s 40 -e 240 -p 40".format(type))
    # os.system("python {}.py --dataset resnet50 -s 40 -e 240 -p 40".format(type))

# os.system("python nn.py --dataset mnist -s 4 -e 24 -p 4")
# os.system("python nn.py --dataset fmnist -s 10 -e 60 -p 10")
# os.system("python nn.py --dataset cifar10 -s 40 -e 240 -p 40")
# os.system("python nn.py --dataset resnet50 -s 40 -e 240 -p 40")
#
# os.system("python boundary.py --dataset mnist -s 4 -e 24 -p 4")
# os.system("python boundary.py --dataset fmnist -s 10 -e 60 -p 10")
# os.system("python boundary.py --dataset cifar10 -s 40 -e 240 -p 40")
# os.system("python boundary.py --dataset resnet50 -s 40 -e 240 -p 40")
#
# os.system("python inv_accu.py --dataset mnist -s 4 -e 24 -p 4")
# os.system("python inv_accu.py --dataset fmnist -s 10 -e 60 -p 10")
# os.system("python inv_accu.py --dataset cifar10 -s 40 -e 240 -p 40")
# os.system("python inv_accu.py --dataset resnet50 -s 40 -e 240 -p 40")
#
# os.system("python inv_conf.py --dataset mnist -s 4 -e 24 -p 4")
# os.system("python inv_conf.py --dataset fmnist -s 10 -e 60 -p 10")
# os.system("python inv_conf.py --dataset cifar10 -s 40 -e 240 -p 40")
# os.system("python inv_conf.py --dataset resnet50 -s 40 -e 240 -p 40")