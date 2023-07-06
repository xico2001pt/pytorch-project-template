import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import yaml
import torch
import torchvision
import torchvision.transforms as transforms
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

    train_dataset = Dataset1('data/', train=True, transform=data_aug)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    test_dataset = Dataset1('data/', train=False, transform=data_aug)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    import torch.optim as optim

    model = Model1(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(model, optimizer, criterion, device=device, log_path='logs/')

    trainer.train(train_loader, test_loader, 2)

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))

    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))


if __name__ == "__main__":
    main()
