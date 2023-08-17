import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
import os


class Dataset1(datasets.CIFAR10):
    def __init__(self, data_dir, train=True, download=True):
        root = os.path.join(sys.path[-1], data_dir)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        super(Dataset1, self).__init__(
            root=root, train=train, transform=transform, download=download
        )
