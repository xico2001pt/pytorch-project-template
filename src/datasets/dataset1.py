import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Dataset1(datasets.CIFAR10):
    def __init__(self, root_dir, train=True, download=True):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        super(Dataset1, self).__init__(
            root=root_dir, train=train, transform=transform, download=download
        )
