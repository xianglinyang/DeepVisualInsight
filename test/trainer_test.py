import unittest
from deepvisualinsight.MMS import MMS
import sys


class MyTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MyTestCase, self).__init__(*args, **kwargs)

        content_path = "E:\\DVI_exp_data\\resnet18_cifar10"
        sys.path.append(content_path)

        from Model.model import resnet18
        net = resnet18()
        # net = ResNet18()
        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        self.mms = MMS(content_path, net, 0, 200, 1, 512, 10, classes, cmap="tab10", resolution=100,
                       neurons=256, verbose=1, temporal=False, split=-1, advance_border_gen=True,
                       attack_device="cuda:0")

    ################################################ Trainer ######################################################
    # mms.data_preprocessing()
    # mms.visualization_for_all()


if __name__ == '__main__':
    unittest.main()
