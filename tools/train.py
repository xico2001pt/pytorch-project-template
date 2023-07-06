import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from models.model1 import Model1
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from core.trainer import Trainer
from utils.loader import load_config, load_dataset, load_optimizer

def main():
    # TODO: Add argparse
    config = load_config('configs/config.yaml')

    trainig_config = config['training']

    dataset_name = trainig_config['dataset']
    epochs = trainig_config['epochs']
    num_workers = trainig_config['num_workers']
    batch_size = trainig_config['batch_size']
    #train_split = trainig_config['train_split']
    optimizer = trainig_config['optimizer']
    #loss = trainig_config['loss']
    #metrics = trainig_config['metrics']
    #stop_condition = trainig_config['stop_condition']

    dataset = load_dataset(dataset_name)

    train_dataset = Subset(dataset, range(5000, len(dataset)))
    validation_dataset = Subset(dataset, range(5000))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    

    metrics = {
        "Accuracy": lambda y_pred, y_true: (y_pred.argmax(dim=1) == y_true).float().mean()
    }

    import torch.optim as optim

    model = Model1(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = load_optimizer(optimizer, model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model.to(device)

    # TODO: Create log folder with timestamp
    trainer = Trainer(model, optimizer, criterion, device=device, log_path='logs/')

    trainer.train(train_loader, validation_loader, epochs, metrics=metrics)

    print("Finished Training")


if __name__ == "__main__":
    main()
