import os
import sys
import time
import torch
import argparse
from datetime import datetime
from torch.utils.data import DataLoader


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


from src.trainers.trainer import Trainer
from src.utils.loader import Loader
from src.utils.logger import Logger

CONFIGS_DIR = os.path.join(BASE_DIR, "configs")  # TODO: Move to config
LOGS_DIR = os.path.join(BASE_DIR, "logs")


def _load_train_data(loader, training_config, model):
    dataset_name = training_config["dataset"]
    optimizer_name = training_config["optimizer"]
    loss_name = training_config["loss"]
    metrics_names = training_config["metrics"]
    scheduler_name = training_config["scheduler"]
    stop_condition_name = training_config["stop_condition"]
    hyperparameters = training_config["hyperparameters"]

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
        "epochs": hyperparameters["epochs"],
        "num_workers": hyperparameters["num_workers"],
        "batch_size": hyperparameters["batch_size"],
        "train_val_split": hyperparameters["train_val_split"],
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


def _log_hyperparameters(logger, hyperparameters):
    logger.log_yaml("Loading hyperparameters", hyperparameters)


def main(args):
    log_dir = os.path.join(LOGS_DIR, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logger = Logger(log_dir, verbose=True)

    try:
        loader = Loader(CONFIGS_DIR, logger)

        logger.info("Loading configuration files...")

        config = loader.load_config_file(args.config)

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

        _log_hyperparameters(logger, training_config["hyperparameters"])

        train_loader, validation_loader = _get_dataloaders(dataset, batch_size, num_workers, train_val_split)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {device}")

        model.to(device)

        trainer = Trainer(model, loss, device=device, logger=logger)

        start_time = time.time()
        
        trainer.train(
            train_loader,
            validation_loader,
            epochs,
            optimizer,
            scheduler=scheduler,
            stop_condition=stop_condition,
            metrics=metrics,
        )

        end_time = time.time()

        logger.info(f"Training took {end_time - start_time} seconds")

        logger.info("Training finished")

    except Exception:
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = args.parse_args()

    main(args)
