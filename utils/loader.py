import yaml
import os
from glob import glob
import importlib
from core.losses import *
from core.optimizers import *

def load_modules(path : str) -> list:
    modules = []
    for file in glob(os.path.join(path, "*.py")):
        name = os.path.splitext(os.path.basename(file))[0]
        name = f"{path}.{name}"
        modules.append(importlib.import_module(name))
    return modules

dataset_modules = load_modules('datasets')

def load_config(path : str) -> dict:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_dataset(name : str):
    datasets_config = load_config('configs/datasets.yaml')
    dataset_config = datasets_config[name]
    dataset_class = dataset_config['class']
    dataset_args = dataset_config['args']

    dataset_obj = None
    for module in dataset_modules:
        if hasattr(module, dataset_class):
            dataset_obj = getattr(module, dataset_class)
            return dataset_obj(**dataset_args)
    return None

def load_loss(name : str):
    losses_config = load_config('configs/losses.yaml')
    loss_config = losses_config[name]
    loss_class = loss_config['class']
    loss_args = loss_config['args']

    return globals()[loss_class](**loss_args)

def load_optimizer(name : str, model):
    optimizers_config = load_config('configs/optimizers.yaml')
    optimizer_config = optimizers_config[name]
    optimizer_class = optimizer_config['class']
    optimizer_args = optimizer_config['args']

    return globals()[optimizer_class](model.parameters(), **optimizer_args)
