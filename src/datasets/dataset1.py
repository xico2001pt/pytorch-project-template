import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ..utils.utils import process_data_path


class Dataset1(datasets.CIFAR10):
    def __init__(self, data_dir, train=True, download=True):
        root = process_data_path(data_dir)
        print(root)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        super(Dataset1, self).__init__(
            root=root, train=train, transform=transform, download=download
        )
