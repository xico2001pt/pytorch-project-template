import yaml
import os
import src.core as core
import src.datasets as datasets
import src.models as models


class Loader:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir

    def _load_config(self, path: str, name: str, module: list, custom_args: dict = {}):
        configs = self.load_config_file(path)
        config = configs[name]
        config_class = config["class"]
        config_args = config["args"]

        for c in module:
            if c.__name__ == config_class:
                return c(**custom_args, **config_args)
        return None

    def load_config_file(self, path: str) -> dict:
        path = os.path.join(self.config_dir, path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_dataset(self, name: str):
        return self._load_config("datasets.yaml", name, datasets.classes)

    def load_model(self, name: str):
        return self._load_config("models.yaml", name, models.classes)

    def load_loss(self, name: str):
        return self._load_config("losses.yaml", name, core.classes["losses"])

    def load_metrics(self, names: list):
        metrics_dict = {}
        for name in names:
            metrics_dict[name] = self._load_config("metrics.yaml", name, core.classes["metrics"])
        return metrics_dict

    def load_optimizer(self, name: str, model):
        return self._load_config("optimizers.yaml", name, core.classes["optimizers"], {"params": model.parameters()})

    def load_scheduler(self, name: str, optimizer):
        if name == "None":
            return None
        return self._load_config("schedulers.yaml", name, core.classes["schedulers"], {"optimizer": optimizer})

    def load_stop_condition(self, name: str):
        if name == "None":
            return None
        return self._load_config("stop_conditions.yaml", name, core.classes["stop_conditions"])
