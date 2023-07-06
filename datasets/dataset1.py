import torchvision.datasets as datasets
from torch.utils.data import Dataset

class Dataset1(datasets.CIFAR10):
    def __init__(self, root_dir, train=True, transform=None):
        super(Dataset1, self).__init__(root=root_dir, train=train, transform=transform, download=True)

