# PyTorch Project Template

Modular and well-structured template for PyTorch deep learning projects.

If you use this template, please make sure to reference the author by including a link to this repository and the author's GitHub profile.

## Table Of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configurations](#configurations)
- [Logging](#logging)
- [Documentation](#documentation)
- [Future Work](#future-work)
- [Projects Using This Template](#projects-using-this-template)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

The prerequisites for this project are:

- Python 3.6+

To intall this project, first clone the repository:

```bash
git clone https://github.com/xico2001pt/pytorch-project-template
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started

To adapt this project to your needs, read the [Getting Started](docs/README.md#getting-started) section of the [documentation](docs/README.md).

## Usage

To train a model, run the following command:

```bash
python src/tools/train.py --config configs/config.yaml
```

To test a model, run the following command:

```bash
python src/tools/test.py --model weights/model_name.pth
```

More details about the usage of the scripts can be found on the [documentation](docs/README.md).

## Project Structure

```python
pytorch-project-template/
├── configs/  # holds the configuration files
│   ├── config.yaml
│   ├── datasets.yaml
│   ├── losses.yaml
│   ├── metrics.yaml
│   ├── models.yaml
│   ├── optimizers.yaml
│   ├── schedulers.yaml
│   ├── stop_conditions.yaml
│   └── ...
├── data/  # default directory for storing input data
│   └── ...
├── docs/  # documentation files
│   └── ...
├── logs/  # default directory for storing logs
│   ├── YYYY-MM-DD-HH-MM-SS/  # training logs
│   │   ├── checkpoints/  # model checkpoints
│   │   │   ├── best_checkpoint.pth  # best model checkpoint
│   │   │   └── latest_checkpoint.pth  # latest model checkpoint
│   │   ├── log.yaml  # log containing all the information about the run
│   │   └── output.log  # log containing the redirected output of the console
│   └── ...
│── src/  # source code
│   ├── core/  # contains the core functionalities
│   │   ├── __init__.py  # exports the core functionalities
│   │   ├── losses.py  # loss functions
│   │   ├── metrics.py  # evaluation metrics
│   │   ├── optimizers.py  # optimizers
│   │   ├── schedulers.py  # learning rate schedulers
│   │   ├── stop_conditions.py  # stop conditions
│   │   └── ...
│   ├── datasets/  # contains the dataset definitions
│   │   ├── __init__.py  # exports the dataset definitions
│   │   ├── dataset1.py
│   │   └── ...
│   ├── models/  # contains the model definitions
│   │   ├── __init__.py  # exports the model definitions
│   │   ├── model1.py
│   │   └── ...
│   ├── tools/  # scripts for training, testing, etc.
│   │   ├── deploy.py
│   │   ├── visualize.py
│   │   ├── test.py
│   │   ├── train.py
│   │   └── ...
│   ├── trainers/  # contains the trainer definitions
│   │   ├── trainer.py
│   │   └── ...
│   ├── utils/  # utility functions
│   │   ├── constants.py  # contains the constants used in the project
│   │   ├── loader.py  # utility class for loading data using the configurations
│   │   ├── logger.py  # utility class for logging
│   │   ├── utils.py
│   │   └── ...
│   └── ...
├── weights/  # default directory for storing model weights
└── requirements.txt  # project dependencies
```

## Configurations

The configurations are stored in the `configs` directory. The configurations are divided into several files, each one containing the configurations for a specific section of the project. The following files are available:

- `config.yaml`: contains the general configurations for the project
- `datasets.yaml`: contains the configurations for the datasets
- `losses.yaml`: contains the configurations for the loss functions
- `metrics.yaml`: contains the configurations for the evaluation metrics
- `models.yaml`: contains the configurations for the models
- `optimizers.yaml`: contains the configurations for the optimizers
- `schedulers.yaml`: contains the configurations for the learning rate schedulers
- `stop_conditions.yaml`: contains the configurations for the stop conditions

More details about the configurations can be found on the [documentation](docs/README.md).

## Logging

The `Logger` class is used to log all the information about the run. Each run is stored in a directory inside the `logs` directory, where the name of the directory is the date and time of the run. Inside the directory, there are two files:

- `log.yaml`: contains all the information about the run
- `output.log`: contains the redirected output of the console

More details about the logging can be found on the [documentation](docs/README.md).

## Documentation

Read the [documentation](docs/README.md) for more details about the project and all the sections mentioned above.

## Future Work

Some of the features that I would like to add to this project template are:

- [ ] Display model summary
- [ ] Add confusion matrix metric and visualization
- [ ] Add support for distributed training
- [ ] Add support for multi-GPU training

## Projects Using This Template

- Comparative Study on Self-Supervision Methods for Autonomous Driving (TBD...) (2024) by [Francisco Cerqueira](https://github.com/xico2001pt)

If you want to see your project here, please contact the author.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Please **ACKNOWLEDGE THE AUTHOR** if you use this project in your research.

## Acknowledgments

This repository was developed by [Francisco Cerqueira](https://github.com/xico2001pt) and was inspired by the following projects:

- [pytorch-template](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque)
- [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Salem](https://github.com/MrGemy95)
- [YOLOP](https://github.com/hustvl/YOLOP) by [hustvl](https://github.com/hustvl)
