import yaml
import os
import src.core as core
import src.datasets as datasets
import src.models as models
from src.utils.logger import Logger

# Config files paths relative to config_dir
DATASETS_CONFIG_PATH = "datasets.yaml"
MODELS_CONFIG_PATH = "models.yaml"
LOSSES_CONFIG_PATH = "losses.yaml"
METRICS_CONFIG_PATH = "metrics.yaml"
OPTIMIZERS_CONFIG_PATH = "optimizers.yaml"
SCHEDULERS_CONFIG_PATH = "schedulers.yaml"
STOP_CONDITIONS_CONFIG_PATH = "stop_conditions.yaml"


class Loader:
    def __init__(self, config_dir: str, logger: Logger):
        self.config_dir = config_dir
        self.logger = logger

    def _log_config(self, config: dict, name: str):
        title = f"Loading {name} configuration"
        self.logger.log_yaml(title, config)

    def _load_config(self, path: str, name: str, module: list, custom_args: dict = {}):
        configs = self.load_config_file(path)
        config = configs[name]
        config_class = config["class"]
        config_args = config["args"]

        for c in module:
            if c.__name__ == config_class:
                self._log_config(config, name)
                return c(**custom_args, **config_args)
        return None

    def load_config_file(self, path: str) -> dict:
        path = os.path.join(self.config_dir, path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_dataset(self, name: str):
        return self._load_config(DATASETS_CONFIG_PATH, name, datasets.classes)

    def load_model(self, name: str):
        return self._load_config(MODELS_CONFIG_PATH, name, models.classes)

    def load_loss(self, name: str):
        return self._load_config(LOSSES_CONFIG_PATH, name, core.classes["losses"])

    def load_metrics(self, names: list):
        metrics_dict = {}
        for name in names:
            metrics_dict[name] = self._load_config(METRICS_CONFIG_PATH, name, core.classes["metrics"])
        return metrics_dict

    def load_optimizer(self, name: str, model):
        return self._load_config(
            OPTIMIZERS_CONFIG_PATH, name, core.classes["optimizers"], {"params": model.parameters()}
        )

    def load_scheduler(self, name: str, optimizer):
        if name == "None":
            return None
        return self._load_config(SCHEDULERS_CONFIG_PATH, name, core.classes["schedulers"], {"optimizer": optimizer})

    def load_stop_condition(self, name: str):
        if name == "None":
            return None
        return self._load_config(STOP_CONDITIONS_CONFIG_PATH, name, core.classes["stop_conditions"])
