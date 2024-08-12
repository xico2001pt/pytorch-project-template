# Configuration Files
This page contains documentation about the configuration files used in the project.

## Table of Contents
- [Global Configuration Files](#global-configuration-files)
    - [Structure Organization](#structure-organization)
    - [File Structure](#file-structure)
    - [Example Configuration File](#example-configuration-file)
- [Experiment Configuration File](#experiment-configuration-file)
    - [File Structure](#file-structure-1)
        - [Supervised Learning](#supervised-learning)
        - [Test](#test)
    - [Example Configuration File](#example-configuration-file-1)

## Global Configuration Files
Global configuration files contain configurations for a specific part of the project, such as datasets, models, or methods. **The global configuration files are used to define the available options for the experiments.**

### Structure Organization
The global configuration files are stored in the `configs/` directory. The available global configuration files are:

- `datasets.yml`: Contains the configuration for the datasets.
- `losses.yml`: Contains the configuration for the losses.
- `metrics.yml`: Contains the configuration for the metrics.
- `models.yml`: Contains the configuration for the models.
- `optimizers.yml`: Contains the configuration for the optimizers.
- `schedulers.yml`: Contains the configuration for the schedulers.
- `stop_conditions.yml`: Contains the configuration for the stopping conditions.

### File Structure
All of the global configuration files have the same structure. The configuration files are organized as a dictionary with the following keys:

- `class`: The class name associated with the configuration.
- `args`: The arguments required to instantiate the class, which will be passed as to the class constructor.

### Example Configuration File
Here is an entry from the `models.yml` configuration file:

```yaml
cifar10_wideresnet28_2:
  class: WideResNet
  args:
    num_classes: 10
    depth: 28
    width: 2
```

## Experiment Configuration File
Experiment configuration files contain the configuration parameters for the experiments, such as the dataset, model, method, and hyperparameters used in the experiment. Therefore, **the experiment configuration files are used to define the parameters for a specific experiment**.

### File Structure
Independent of the learning paradigm, the experiment configuration files will always contain the following keys:

- `name`: The name of the experiment. It is used to name the log folder and the saved weights. If `null`, the name is the current date and time.
- `model`: The name of the model used in the experiment.

#### Supervised Learning
The experiment configuration files for supervised learning are stored in the `experiments/sl/` directory.

The configuration parameters should be defined inside the `sl_train` key and are the following:

- `train_dataset`: The name of the dataset used for training. When loading the dataset, the constructor parameter `split=train` will be **automatically** passed.
- `val_dataset`: The name of the dataset used for validation. When loading the dataset, the constructor parameter `split=val` will be **automatically** passed.
- `optimizer`: The name of the optimizer used for training.
- `loss`: The name of the loss used in the training phase.
- `metrics`: A list of the metrics used in the training phase.
- `scheduler`: The name of the scheduler used for training. Can be `null`.
- `stop_condition`: The name of the stopping condition used for training. Can be `null`.
- `hyperparameters`: The hyperparameters used in the experiment. This key must contain the following keys:
    - `epochs`: The number of epochs used in the training.
    - `num_workers`: The number of workers used in the data loader.
    - `batch_size`: The batch size used in the data loader.
    - `save_freq`: The frequency of saving the model weights (i.e., save the weights every `save_freq` epochs). If 0, only the best model weights will be saved.

#### Test
The experiment configuration files for testing are usually stored in the same file as the training configuration file.

The configuration parameters should be defined inside the `test` key and are the following:

- `test_dataset`: The name of the dataset used for testing. When loading the dataset, the constructor parameter `split=test` will be **automatically** passed.
- `model_weights_path`: The path to the model weights file, relative to the `weights/` directory.
- `loss`: The name of the loss used in the testing phase.
- `metrics`: A list of the metrics used in the testing phase.
- `hyperparameters`: The hyperparameters used in the experiment. This key must contain the following keys:
    - `num_workers`: The number of workers used in the data loader.
    - `batch_size`: The batch size used in the data loader.

### Example Configuration File
Here is an example of an experiment configuration file:

```yaml
name: cifar10  # Name of experiment (used for log folder and model naming). If null, the name is the current date and time

model: model1  # Name of model

train:
  train_dataset: cifar10  # Dataset to use for training
  val_dataset: cifar10  # Dataset to use for validation
  optimizer: sgd  # Optimizer
  loss: cross_entropy  # Loss function
  metrics: [accuracy]  # Metrics to evaluate
  scheduler: null  # Learning rate scheduler. Can be null
  stop_condition: null  # Condition to stop training. Can be null
  hyperparameters:  # Hyperparameters for training
    epochs: 20  # Number of training epochs
    num_workers: 4  # Number of workers for data loading
    batch_size: 32  # Training batch size
    save_freq: 10  # Save model every save_freq epochs. If 0, only the best model is saved

test:
  test_dataset: cifar10  # Dataset to use for testing
  model_weights_path: cifar10.pth  # Path to model weights (relative to weights folder). If null, config name is used
  loss: cross_entropy  # Loss function
  metrics: [accuracy]  # Metrics to evaluate
  hyperparameters:  # Hyperparameters for testing
    num_workers: 4  # Number of workers for data loading
    batch_size: 32  # Test batch size
```