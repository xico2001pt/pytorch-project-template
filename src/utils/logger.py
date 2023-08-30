import logging
import os
import sys
import yaml


LOG_FILE_NAME = "log.txt"


class Logger:
    def __init__(self, log_dir: str, verbose: bool = True):
        self.log_dir = log_dir

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self._create_log_dir()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(os.path.join(self.log_dir, LOG_FILE_NAME))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if verbose:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def _create_log_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def get_log_dir(self):
        return self.log_dir

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def log_yaml(self, title: str, yaml_dict: dict):
        self.info(title + "\n" + yaml.dump(yaml_dict))
