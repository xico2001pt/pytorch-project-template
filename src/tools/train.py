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

DATASET_CONFIG_NAME = "dataset"
OPTIMIZER_CONFIG_NAME = "optimizer"
LOSS_CONFIG_NAME = "loss"
METRICS_CONFIG_NAME = "metrics"
SCHEDULER_CONFIG_NAME = "scheduler"
STOP_CONDITION_CONFIG_NAME = "stop_condition"
HYPERPARAMETERS_CONFIG_NAME = "hyperparameters"

EPOCHS_CONFIG_NAME = "epochs"
NUM_WORKERS_CONFIG_NAME = "num_workers"
BATCH_SIZE_CONFIG_NAME = "batch_size"
TRAIN_VAL_SPLIT_CONFIG_NAME = "train_val_split"


def _load_train_data(loader, training_config, model, logger):
    dataset_name = training_config[DATASET_CONFIG_NAME]
    optimizer_name = training_config[OPTIMIZER_CONFIG_NAME]
    loss_name = training_config[LOSS_CONFIG_NAME]
    metrics_names = training_config[METRICS_CONFIG_NAME]
    scheduler_name = training_config[SCHEDULER_CONFIG_NAME]
    stop_condition_name = training_config[STOP_CONDITION_CONFIG_NAME]

    dataset, dataset_config = loader.load_dataset(dataset_name)
    logger.log_config(DATASET_CONFIG_NAME, dataset_config)

    optimizer, optimizer_config = loader.load_optimizer(optimizer_name, model)
    logger.log_config(OPTIMIZER_CONFIG_NAME, optimizer_config)

    loss, loss_config = loader.load_loss(loss_name)
    logger.log_config(LOSS_CONFIG_NAME, loss_config)

    metrics, metrics_config = loader.load_metrics(metrics_names)
    metrics_dict = {metric_name: metrics_config[metric_name] for metric_name in metrics_names}
    logger.log_config(METRICS_CONFIG_NAME, metrics_dict)

    scheduler, scheduler_config = loader.load_scheduler(scheduler_name, optimizer)
    logger.log_config(SCHEDULER_CONFIG_NAME, scheduler_config)

    stop_condition, stop_condition_config = loader.load_stop_condition(stop_condition_name)
    logger.log_config(STOP_CONDITION_CONFIG_NAME, stop_condition_config)

    hyperparameters = training_config[HYPERPARAMETERS_CONFIG_NAME]
    logger.log_config(HYPERPARAMETERS_CONFIG_NAME, hyperparameters)

    return {
        DATASET_CONFIG_NAME: dataset,
        OPTIMIZER_CONFIG_NAME: optimizer,
        LOSS_CONFIG_NAME: loss,
        METRICS_CONFIG_NAME: metrics,
        SCHEDULER_CONFIG_NAME: scheduler,
        STOP_CONDITION_CONFIG_NAME: stop_condition,
        HYPERPARAMETERS_CONFIG_NAME: hyperparameters,
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


def _get_device(logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log_device(device)
    device = torch.device(device)
    return device


def _log_training_time(start_time, end_time, logger):
    logger.log_time("Training", end_time - start_time)


def main(args):
    log_dir = os.path.join(LOGS_DIR, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logger = Logger(log_dir, console_output=True, file_output=True)

    try:
        loader = Loader(CONFIGS_DIR)

        logger.info("Loading configuration files...")

        config = loader.load_config_file(args.config)

        model, model_config = loader.load_model(config["model"])
        # TODO: log summary

        training_config = config["train"]
        data = _load_train_data(loader, training_config, model, logger)
        (
            dataset,
            optimizer,
            loss,
            metrics,
            scheduler,
            stop_condition,
            hyperparameters,
        ) = data.values()

        epochs, num_workers, batch_size, train_val_split = hyperparameters.values()

        train_loader, validation_loader = _get_dataloaders(dataset, batch_size, num_workers, train_val_split)

        device = _get_device(logger)

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

        _log_training_time(start_time, end_time, logger)

        logger.save_log()

    except Exception:
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = args.parse_args()

    main(args)
