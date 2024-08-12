# PyTorch Project Template

Modular and well-structured template for PyTorch deep learning projects.

If you use this template, please make sure to reference the author by including a link to this repository and the author's GitHub profile.

## Table Of Contents

- [License](#license)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configurations](#configurations)
- [Logging](#logging)
- [Documentation](#documentation)
- [Future Work](#future-work)
- [Projects Using This Template](#projects-using-this-template)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Please **ACKNOWLEDGE THE AUTHOR** if you use this template in your project by including a link to this repository and the author's GitHub profile.

## Installation

The prerequisites for this project are:

- Python 3.6+
- pip
- git

To intall this project, first clone the repository:

```bash
git clone https://github.com/xico2001pt/pytorch-project-template
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started

To adapt this project to your needs, it is recommended to read the [documentation](docs/README.md) file. This file contains a brief overview of the available documentation and links to the different sections of the documentation.

## Usage

To train a model, run the following command:

```bash
python -m src.tools.train --config config.yaml
```

To test a model, run the following command:

```bash
python -m src.tools.test --config config.yaml
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
│   └── stop_conditions.yaml
├── data/  # default directory for storing input data
├── docs/  # documentation files
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
│   │   └── stop_conditions.py  # stop conditions
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

- `config.yaml`: contains the configurations for a specific run (can be given any name, as it is passed as an argument to the scripts)
- `datasets.yaml`: contains the configurations for the datasets
- `losses.yaml`: contains the configurations for the loss functions
- `metrics.yaml`: contains the configurations for the evaluation metrics
- `models.yaml`: contains the configurations for the models
- `optimizers.yaml`: contains the configurations for the optimizers
- `schedulers.yaml`: contains the configurations for the learning rate schedulers
- `stop_conditions.yaml`: contains the configurations for the stop conditions

More details about the configurations can be found on the [documentation](docs/README.md).

## Logging

The `Logger` class is used to log all the information about the run. Each run is stored in a directory inside the `logs` directory, where the name of the directory is the date and time of the run. Inside the directory, the following files can exist:

- `train_log.yaml`: contains all the information about the training run
- `train_output.log`: contains the redirected output of the console during training
- `test_log.yaml`: contains all the information about the testing run
- `test_output.log`: contains the redirected output of the console during testing

## Documentation

Read the [documentation](docs/README.md) for more details about the project and all the sections mentioned above.

## Future Work

Some of the features that I would like to add to this project template are:

- [ ] Display model summary
- [ ] Load pre-trained weights
- [ ] Add support for inlining the configurations
- [ ] Log multi-term losses
- [ ] Add support for distributed training

## Projects Using This Template

- [Exploring Label Efficiency with Semi-Supervision and Self-Supervision Methods](https://github.com/xico2001pt/exploring-label-efficiency) (2024) by [Francisco Cerqueira](https://github.com/xico2001pt)

If you want to see your project here, please contact the author or create a pull request.

## Contributing

If you want to contribute to this project, please contact the author or create a pull request with a description of the feature or bug fix.

## Acknowledgments

This repository was developed by [Francisco Cerqueira](https://github.com/xico2001pt) and was inspired by the following projects:

- [pytorch-template](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque)
- [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Salem](https://github.com/MrGemy95)
- [YOLOP](https://github.com/hustvl/YOLOP) by [hustvl](https://github.com/hustvl)
