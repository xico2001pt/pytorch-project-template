import yaml
import os
from glob import glob
import importlib
from src.core.losses import *
from src.core.metrics import *
from src.core.optimizers import *
from src.core.schedulers import *
from src.core.stop_conditions import *


def load_modules(path: str) -> list:
    modules = []
    items = os.listdir(path)
    # Path with dots
    dot_path = path.replace("/", ".")

    for item in items:
        item_path = os.path.join(path, item)

        if os.path.isdir(item_path):
            modules.extend(load_modules(item_path))
        elif os.path.isfile(item_path) and item.endswith('.py'):
            module_name = os.path.splitext(item)[0]
            module = importlib.import_module(f"{dot_path}.{module_name}")
            modules.append(module)

    return modules


dataset_modules = load_modules("src/datasets")
model_modules = load_modules("src/models")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(name: str):
    datasets_config = load_config("configs/datasets.yaml")
    dataset_config = datasets_config[name]
    dataset_class = dataset_config["class"]
    dataset_args = dataset_config["args"]

    dataset_obj = None
    for module in dataset_modules:
        if hasattr(module, dataset_class):
            dataset_obj = getattr(module, dataset_class)
            return dataset_obj(**dataset_args)
    return None


def load_model(name: str):
    models_config = load_config("configs/models.yaml")
    model_config = models_config[name]
    model_class = model_config["class"]
    model_args = model_config["args"]

    model_obj = None
    for module in model_modules:
        if hasattr(module, model_class):
            model_obj = getattr(module, model_class)
            return model_obj(**model_args)
    return None


def load_loss(name: str):
    losses_config = load_config("configs/losses.yaml")
    loss_config = losses_config[name]
    loss_class = loss_config["class"]
    loss_args = loss_config["args"]

    return globals()[loss_class](**loss_args)


def load_metrics(names: list):
    metrics_config = load_config("configs/metrics.yaml")
    metrics = {}
    for name in names:
        metric_config = metrics_config[name]
        metric_class = metric_config["class"]
        metric_args = metric_config["args"]

        metrics[name] = globals()[metric_class](**metric_args)
    return metrics


def load_optimizer(name: str, model):
    optimizers_config = load_config("configs/optimizers.yaml")
    optimizer_config = optimizers_config[name]
    optimizer_class = optimizer_config["class"]
    optimizer_args = optimizer_config["args"]

    return globals()[optimizer_class](model.parameters(), **optimizer_args)


def load_scheduler(name: str, optimizer):
    schedulers_config = load_config("configs/schedulers.yaml")
    scheduler_config = schedulers_config[name]
    scheduler_class = scheduler_config["class"]
    scheduler_args = scheduler_config["args"]

    return globals()[scheduler_class](optimizer, **scheduler_args)


def load_stop_condition(name: str):
    stop_conditions_config = load_config("configs/stop_conditions.yaml")
    stop_condition_config = stop_conditions_config[name]
    stop_condition_class = stop_condition_config["class"]
    stop_condition_args = stop_condition_config["args"]

    return globals()[stop_condition_class](**stop_condition_args)