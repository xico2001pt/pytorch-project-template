import torchvision.datasets as datasets
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
