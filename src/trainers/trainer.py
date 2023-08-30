import torch
from tqdm import tqdm
import numpy as np
import json
import os


class Trainer:
    def __init__(self, model, loss_fn, device, logger):
        # TODO: Add configs
        # TODO: Logging missing
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.checkpoints_path = os.path.join(self.logger.get_log_dir(), "checkpoints")

        os.makedirs(self.checkpoints_path, exist_ok=True)

    def _log_epoch_stats(self, loss: list, metrics: dict, split_name: str):
        yaml_dict = {"Loss": loss, "Metrics": metrics}
        self.logger.log_yaml(f"{split_name} Stats", yaml_dict)

    def _save_checkpoint(self, filename: str, optimizer, epoch: int):
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(save_dict, os.path.join(self.checkpoints_path, filename))

    def _save_log(self, obj, filename):
        json.dump(obj, open(os.path.join(self.logger.get_log_dir(), filename), "w"))

    def _compute_loss(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return outputs, loss

    def _epoch_iteration(self, dataloader, is_train=True, optimizer=None, metrics={}, description="Train"):
        if is_train:
            assert optimizer is not None, "optimizer must be provided for training"

        num_batches = len(dataloader)

        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        total_metrics = {metric: 0.0 for metric in metrics}

        with torch.set_grad_enabled(is_train):
            for inputs, targets in tqdm(dataloader, desc=description):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if is_train:
                    optimizer.zero_grad()

                outputs, loss = self._compute_loss(inputs, targets)

                if is_train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

                for metric in metrics:
                    total_metrics[metric] += metrics[metric](outputs, targets).item()

        avg_loss = total_loss / num_batches

        for metric in total_metrics:
            total_metrics[metric] /= num_batches

        return avg_loss, total_metrics

    def train(
        self,
        train_dataloader,
        validation_dataloader,
        num_epochs,
        optimizer,
        scheduler=None,
        stop_condition=None,
        metrics={},
    ):
        print(f"Training for {num_epochs} epochs")

        train_history = {"loss": [], "metrics": {metric: [] for metric in metrics}}
        validation_history = {"loss": [], "metrics": {metric: [] for metric in metrics}}
        best_validation_loss = np.inf

        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")

            train_loss, train_metrics = self._epoch_iteration(
                train_dataloader,
                is_train=True,
                optimizer=optimizer,
                metrics=metrics,
                description="Train",
            )

            self._log_epoch_stats(train_loss, train_metrics, "Train")

            validation_loss, validation_metrics = self._epoch_iteration(
                validation_dataloader,
                is_train=False,
                metrics=metrics,
                description="Validation",
            )

            self._log_epoch_stats(validation_loss, validation_metrics, "Validation")

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self._save_checkpoint("best_checkpoint.pth", optimizer, epoch)

            self._save_checkpoint("latest_checkpoint.pth", optimizer, epoch)

            train_history["loss"].append(train_loss)
            validation_history["loss"].append(validation_loss)

            for metric in metrics:
                train_history["metrics"][metric].append(train_metrics[metric])
                validation_history["metrics"][metric].append(validation_metrics[metric])

            self._save_log(train_history["loss"], "train_losses.json")
            self._save_log(train_history["metrics"], "train_metrics.json")
            self._save_log(validation_history["loss"], "validation_losses.json")
            self._save_log(validation_history["metrics"], "validation_metrics.json")
            # TODO: Save configs

            if stop_condition and stop_condition(train_loss, validation_loss):
                print("Stopping due to stop condition")
                break

            if scheduler:
                scheduler.step()

    def test(self, test_dataloader, metrics={}):
        test_loss, test_metrics = self._epoch_iteration(
            test_dataloader, is_train=False, metrics=metrics, description="Test"
        )

        self._log_epoch_stats(test_loss, test_metrics, "Test")

        self._save_log(test_loss, "test_loss.json")
        self._save_log(test_metrics, "test_metrics.json")
        # TODO: Save configs
