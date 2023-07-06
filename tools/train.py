import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import yaml
import torch
import torchvision
import torchvision.transforms as transforms
# import subset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from models.model1 import Model1
from datasets.dataset1 import Dataset1
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from trainer import Trainer

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('test.png')

def main():
    # TODO: Add argparse
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    data_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 8

    dataset = Dataset1('data/', train=True, transform=data_aug)
    train_dataset = Subset(dataset, range(5000, len(dataset)))
    validation_dataset = Subset(dataset, range(5000))
    test_dataset = Dataset1('data/', train=False, transform=data_aug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    #import f1score

    metrics = {
        "accuracy": lambda y_pred, y_true: (y_pred.argmax(dim=1) == y_true).float().mean()
    }

    import torch.optim as optim

    model = Model1(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model.to(device)

    # TODO: Create log folder with timestamp
    trainer = Trainer(model, optimizer, criterion, device=device, log_path='logs/')

    trainer.train(train_loader, validation_loader, 2, metrics=metrics)

    trainer.test(test_loader, metrics=metrics)


if __name__ == "__main__":
    main()
