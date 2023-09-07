import yaml
import os
import src.core as core
import src.datasets as datasets
import src.models as models
from .constants import Constants as c


class Loader:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir

    def _load_config(self, path: str, name: str, module: list, custom_args: dict = {}) -> tuple:
        configs = self.load_config_file(path)
        config = configs[name]
        config_class = config["class"]
        config_args = config["args"]

        res = None
        for cl in module:
            if cl.__name__ == config_class:
                res = cl(**config_args, **custom_args)
                break
        return res, config

    def load_config_file(self, path: str) -> dict:
        path = os.path.join(self.config_dir, path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_dataset(self, name: str):
        return self._load_config(c.Loader.DATASETS_CONFIG_FILENAME, name, datasets.classes)

    def load_model(self, name: str):
        return self._load_config(c.Loader.MODELS_CONFIG_FILENAME, name, models.classes)

    def load_loss(self, name: str):
        return self._load_config(c.Loader.LOSSES_CONFIG_FILENAME, name, core.classes["losses"])

    def load_metrics(self, names: list):
        metrics_dict = {}
        metrics_configs_dict = {}
        for name in names:
            metrics_dict[name], metrics_configs_dict[name] = self._load_config(
                c.Loader.METRICS_CONFIG_FILENAME, name, core.classes["metrics"]
            )
        return metrics_dict, metrics_configs_dict

    def load_optimizer(self, name: str, model):
        return self._load_config(
            c.Loader.OPTIMIZERS_CONFIG_FILENAME, name, core.classes["optimizers"], {"params": model.parameters()}
        )

    def load_scheduler(self, name: str, optimizer):
        if name == None:
            return None, None
        return self._load_config(
            c.Loader.SCHEDULERS_CONFIG_FILENAME, name, core.classes["schedulers"], {"optimizer": optimizer}
        )

    def load_stop_condition(self, name: str):
        if name == None:
            return None, None
        return self._load_config(c.Loader.STOP_CONDITIONS_CONFIG_FILENAME, name, core.classes["stop_conditions"])
