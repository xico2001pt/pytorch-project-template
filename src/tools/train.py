import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.loader import Loader
from src.trainers.trainer import Trainer
from torch.utils.data import DataLoader
from datetime import datetime
import torch


def _load_train_data(loader, training_config, model):
    dataset_name = training_config["dataset"]
    optimizer_name = training_config["optimizer"]
    loss_name = training_config["loss"]
    metrics_names = training_config["metrics"]
    scheduler_name = training_config["scheduler"]
    stop_condition_name = training_config["stop_condition"]

    dataset = loader.load_dataset(dataset_name)
    optimizer = loader.load_optimizer(optimizer_name, model)
    loss = loader.load_loss(loss_name)
    metrics = loader.load_metrics(metrics_names)
    scheduler = loader.load_scheduler(scheduler_name, optimizer)
    stop_condition = loader.load_stop_condition(stop_condition_name)

    return {
        "dataset": dataset,
        "optimizer": optimizer,
        "loss": loss,
        "metrics": metrics,
        "scheduler": scheduler,
        "stop_condition": stop_condition,
        "epochs": training_config["epochs"],
        "num_workers": training_config["num_workers"],
        "batch_size": training_config["batch_size"],
        "train_val_split": training_config["train_val_split"],
    }


def _get_dataloaders(dataset, batch_size, num_workers, train_val_split):
    train_size = int(train_val_split * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader


def main():
    # TODO: Add argparse
    loader = Loader(BASE_DIR)

    config = loader.load_config_file("configs/config.yaml")

    model = loader.load_model(config["model"])

    training_config = config["train"]
    data = _load_train_data(loader, training_config, model)
    (
        dataset,
        optimizer,
        loss,
        metrics,
        scheduler,
        stop_condition,
        epochs,
        num_workers,
        batch_size,
        train_val_split,
    ) = data.values()

    train_loader, validation_loader = _get_dataloaders(dataset, batch_size, num_workers, train_val_split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model.to(device)

    log_path = os.path.join(BASE_DIR, "logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    trainer = Trainer(model, loss, device=device, log_path=log_path)

    trainer.train(
        train_loader,
        validation_loader,
        epochs,
        optimizer,
        scheduler=scheduler,
        stop_condition=stop_condition,
        metrics=metrics,
    )

    print("Finished Training")


if __name__ == "__main__":
    main()
