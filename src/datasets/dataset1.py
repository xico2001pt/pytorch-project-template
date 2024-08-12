import torchvision
import torchvision.transforms as T
import torch.utils.data as torch_data
from ..utils.utils import process_data_path, split_train_val_data


class Dataset1(torch_data.Dataset):
    def __init__(self, data_dir, split='train', download=True, train_val_split=0.9):
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be either 'train', 'val' or 'test'")

        root = process_data_path(data_dir)

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                (0.49139968, 0.48215827, 0.44653124),
                (0.24703233, 0.24348505, 0.26158768)
            )
        ])
        train_or_val = split in ['train', 'val']

        self.dataset = torchvision.datasets.CIFAR10(
            root,
            train=train_or_val,
            transform=transform,
            download=True
        )

        if train_or_val:
            splitted_data = split_train_val_data(self.dataset, train_val_split)
            self.dataset = splitted_data[0] if split == 'train' else splitted_data[1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
