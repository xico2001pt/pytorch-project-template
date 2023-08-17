import yaml
import os
import importlib
import src.core.losses as losses
import src.core.metrics as metrics
import src.core.optimizers as optimizers
import src.core.schedulers as schedulers
import src.core.stop_conditions as stop_conditions


class Loader:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.dataset_modules = self._load_modules("src.datasets")
        self.model_modules = self._load_modules("src.models")

    def _load_modules(self, import_path: str) -> list:
        modules = []
        path = os.path.join(self.base_dir, *import_path.split("."))

        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                modules.extend(self._load_modules(item_path))
            elif os.path.isfile(item_path) and item.endswith(".py"):
                module_name = os.path.splitext(item)[0]
                module = importlib.import_module(f"{import_path}.{module_name}")
                modules.append(module)

        return modules

    def load_config(self, path: str) -> dict:
        path = os.path.join(self.base_dir, path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_dataset(self, name: str):
        datasets_config = self.load_config("configs/datasets.yaml")
        dataset_config = datasets_config[name]
        dataset_class = dataset_config["class"]
        dataset_args = dataset_config["args"]

        dataset_obj = None
        for module in self.dataset_modules:
            if hasattr(module, dataset_class):
                dataset_obj = getattr(module, dataset_class)
                return dataset_obj(**dataset_args)
        return None

    def load_model(self, name: str):
        models_config = self.load_config("configs/models.yaml")
        model_config = models_config[name]
        model_class = model_config["class"]
        model_args = model_config["args"]

        model_obj = None
        for module in self.model_modules:
            if hasattr(module, model_class):
                model_obj = getattr(module, model_class)
                return model_obj(**model_args)
        return None

    def load_loss(self, name: str):
        losses_config = self.load_config("configs/losses.yaml")
        loss_config = losses_config[name]
        loss_class = loss_config["class"]
        loss_args = loss_config["args"]

        return losses.__dict__[loss_class](**loss_args)

    def load_metrics(self, names: list):
        metrics_config = self.load_config("configs/metrics.yaml")
        metrics_dict = {}
        for name in names:
            metric_config = metrics_config[name]
            metric_class = metric_config["class"]
            metric_args = metric_config["args"]

            metrics_dict[name] = metrics.__dict__[metric_class](**metric_args)
        return metrics_dict

    def load_optimizer(self, name: str, model):
        optimizers_config = self.load_config("configs/optimizers.yaml")
        optimizer_config = optimizers_config[name]
        optimizer_class = optimizer_config["class"]
        optimizer_args = optimizer_config["args"]

        return optimizers.__dict__[optimizer_class](model.parameters(), **optimizer_args)

    def load_scheduler(self, name: str, optimizer):
        schedulers_config = self.load_config("configs/schedulers.yaml")
        scheduler_config = schedulers_config[name]
        scheduler_class = scheduler_config["class"]
        scheduler_args = scheduler_config["args"]

        return schedulers.__dict__[scheduler_class](optimizer, **scheduler_args)

    def load_stop_condition(self, name: str):
        stop_conditions_config = self.load_config("configs/stop_conditions.yaml")
        stop_condition_config = stop_conditions_config[name]
        stop_condition_class = stop_condition_config["class"]
        stop_condition_args = stop_condition_config["args"]

        return stop_conditions.__dict__[stop_condition_class](**stop_condition_args)
