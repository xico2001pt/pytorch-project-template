import torch
from tqdm import tqdm
import numpy as np
import os
from ..utils.constants import Constants as c


class Trainer:
    def __init__(self, model, loss_fn, device, logger):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.checkpoints_path = os.path.join(self.logger.get_log_dir(), c.Trainer.Checkpoints.CHECKPOINTS_DIR)

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

    def _save_stats(self, loss_history, metrics_history, name):
        self.logger.add_log_entry(f"{name}_history", {"loss": loss_history, "metrics": metrics_history})

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

    def save_best_model(self, save_dir, filename):
        checkpoint_file = os.path.join(self.checkpoints_path, c.Trainer.Checkpoints.BEST_CHECKPOINT_FILENAME)
        weights = torch.load(checkpoint_file)["model"]
        save_path = os.path.join(save_dir, f"{filename}.pth")
        torch.save(weights, save_path)

        self.logger.info(f"Best model saved to {save_path}")

    def train(
        self,
        train_dataloader,
        validation_dataloader,
        num_epochs,
        optimizer,
        scheduler=None,
        stop_condition=None,
        metrics={},
        start_epoch=1,
        save_freq=1,
    ):
        self.logger.info(f"Training for {num_epochs - start_epoch + 1} epochs")

        train_history = {"loss": [], "metrics": {metric: [] for metric in metrics}}
        validation_history = {"loss": [], "metrics": {metric: [] for metric in metrics}}
        best_validation_loss = np.inf

        for epoch in range(start_epoch, num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{num_epochs}")

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
                self._save_checkpoint(c.Trainer.Checkpoints.BEST_CHECKPOINT_FILENAME, optimizer, epoch)

            if save_freq and (epoch % save_freq == 0 or epoch == num_epochs):
                self._save_checkpoint(c.Trainer.Checkpoints.LATEST_CHECKPOINT_FILENAME, optimizer, epoch)

            train_history["loss"].append(train_loss)
            validation_history["loss"].append(validation_loss)

            for metric in metrics:
                train_history["metrics"][metric].append(train_metrics[metric])
                validation_history["metrics"][metric].append(validation_metrics[metric])

            self._save_stats(train_history["loss"], train_history["metrics"], "train")
            self._save_stats(validation_history["loss"], validation_history["metrics"], "validation")

            if stop_condition and stop_condition(train_loss, validation_loss):
                self.logger.warning("Stopping due to stop condition")
                break

            if scheduler:
                scheduler.step()

        self.logger.info(f"Best validation loss: {best_validation_loss}")

    def test(self, test_dataloader, metrics={}):
        test_loss, test_metrics = self._epoch_iteration(
            test_dataloader, is_train=False, metrics=metrics, description="Test"
        )

        self._log_epoch_stats(test_loss, test_metrics, "Test")

        self._save_stats(test_loss, test_metrics, "test")
