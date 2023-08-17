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

    def _load_config(self, configs: str | dict, name: str, module: dict, custom_args: dict = {}):
        configs = self.load_config_file(configs) if isinstance(configs, str) else configs
        config = configs[name]
        config_class = config["class"]
        config_args = config["args"]

        return module.__dict__[config_class](**custom_args, **config_args)

    def load_config_file(self, path: str) -> dict:
        path = os.path.join(self.base_dir, path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_dataset(self, name: str):
        datasets_config = self.load_config_file("configs/datasets.yaml")
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
        models_config = self.load_config_file("configs/models.yaml")
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
        return self._load_config("configs/losses.yaml", name, losses)

    def load_metrics(self, names: list):
        metrics_config = self.load_config_file("configs/metrics.yaml")
        metrics_dict = {}
        for name in names:
            metrics_dict[name] = self._load_config(metrics_config, name, metrics)
        return metrics_dict

    def load_optimizer(self, name: str, model):
        return self._load_config("configs/optimizers.yaml", name, optimizers, {"params": model.parameters()})

    def load_scheduler(self, name: str, optimizer):
        return self._load_config("configs/schedulers.yaml", name, schedulers, {"optimizer": optimizer})

    def load_stop_condition(self, name: str):
        return self._load_config("configs/stop_conditions.yaml", name, stop_conditions)
