import torch
from tqdm import tqdm
import numpy as np
import json
import os

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, log_path):
        # TODO: Add configs
        # TODO: Add scheduler
        # TODO: Add early stopping
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.log_path = log_path
        self.checkpoints_path = os.path.join(log_path, "checkpoints")

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
    
    @staticmethod
    def _print_epoch_stats(loss, metrics : dict, split_name : str):
        print(f"{split_name} Loss: {loss:.4f}")
        for metric in metrics:
            print(f"\t{split_name} {metric}: {metrics[metric]:.4f}")
    
    def _save_checkpoint(self, filename, epoch):
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(save_dict, os.path.join(self.checkpoints_path, filename))
    
    def _save_log(self, obj, filename):
        json.dump(obj, open(os.path.join(self.log_path, filename), "w"))

    def _epoch_iteration(self, dataloader, is_train=True, metrics={}):
        if is_train:
            assert self.optimizer is not None, "optimizer must be provided for training"
        
        num_batches = len(dataloader)

        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        total_metrics = {metric: 0.0 for metric in metrics}

        with torch.set_grad_enabled(is_train):
            for inputs, targets in tqdm(dataloader, desc=f"{'Train' if is_train else 'Validation'}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                for metric in metrics:
                    total_metrics[metric] += metrics[metric](outputs, targets).item()

        avg_loss = total_loss / num_batches

        for metric in total_metrics:
            total_metrics[metric] /= num_batches

        return avg_loss, total_metrics
    
    def train(self, train_dataloader, validation_dataloader, num_epochs, metrics={}):
        print(f"Training for {num_epochs} epochs")

        train_history = {"loss": [], "metrics": {metric: [] for metric in metrics}}
        validation_history = {"loss": [], "metrics": {metric: [] for metric in metrics}}
        best_validation_loss = np.inf

        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")

            train_loss, train_metrics = self._epoch_iteration(
                train_dataloader, 
                is_train=True, 
                metrics=metrics
            )

            # TODO: If verbose
            Trainer._print_epoch_stats(train_loss, train_metrics, "Train")

            validation_loss, validation_metrics = self._epoch_iteration(
                validation_dataloader, 
                is_train=False, 
                metrics=metrics
            )

            Trainer._print_epoch_stats(validation_loss, validation_metrics, "Validation")

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self._save_checkpoint("best_checkpoint.pth", epoch)

            self._save_checkpoint("latest_checkpoint.pth", epoch)

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
            
            print("Finished Training")
            
    def test(self, test_dataloader, metrics={}):
        test_loss, test_metrics = self._epoch_iteration(
            test_dataloader,
            is_train=False, 
            metrics=metrics
        )

        Trainer._print_epoch_stats(test_loss, test_metrics, "Test")

        self._save_log(test_loss, "test_loss.json")
        self._save_log(test_metrics, "test_metrics.json")
        # TODO: Save configs
