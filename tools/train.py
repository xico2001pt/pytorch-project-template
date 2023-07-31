import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from datetime import datetime
import torch
from torch.utils.data import DataLoader, Subset
from models.model1 import Model1
from trainers.trainer import Trainer
from utils.loader import load_config, load_dataset, load_loss, load_optimizer, load_scheduler, load_stop_condition

def main():
    # TODO: Add argparse
    config = load_config('configs/config.yaml')

    trainig_config = config['training']

    # TODO: One time use should be inlined
    dataset_name = trainig_config['dataset']
    epochs = trainig_config['epochs']
    num_workers = trainig_config['num_workers']
    batch_size = trainig_config['batch_size']
    #train_split = trainig_config['train_split']
    optimizer = trainig_config['optimizer']
    loss = trainig_config['loss']
    #metrics = trainig_config['metrics']
    scheduler = trainig_config['scheduler']
    stop_condition = trainig_config['stop_condition']

    dataset = load_dataset(dataset_name)

    train_dataset = Subset(dataset, range(5000, len(dataset)))
    validation_dataset = Subset(dataset, range(5000))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    

    metrics = {
        "Accuracy": lambda y_pred, y_true: (y_pred.argmax(dim=1) == y_true).float().mean()
    }

    model = Model1(num_classes=10)
    loss = load_loss(loss)
    optimizer = load_optimizer(optimizer, model)
    scheduler = None if scheduler == "None" else load_scheduler(scheduler, optimizer)
    stop_condition = None if stop_condition == "None" else load_stop_condition(stop_condition)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model.to(device)

    log_path = os.path.join('logs', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    trainer = Trainer(model, optimizer, loss, scheduler, stop_condition, device=device, log_path=log_path)

    trainer.train(train_loader, validation_loader, epochs, metrics=metrics)

    print("Finished Training")


if __name__ == "__main__":
    main()
