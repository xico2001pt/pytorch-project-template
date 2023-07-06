# PyTorch Project Template

PyTorch template for deep learning projects.

## Table Of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configurations](#configurations)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

To intall this project, first clone the repository:

```bash
git clone https://github.com/xico2001pt/pytorch-project-template
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

## Project Structure

```python
pytorch-project-template/
│
├── configs/  # holds the configuration files
│   ├── config.yaml
│   ├── datasets.yaml
│   ├── losses.yaml
│   ├── metrics.yaml
│   └── ...
│
├── core/  # contains the core functionality
│   ├── losses.py  # loss functions
│   ├── metrics.py  # evaluation metrics
│   ├── optimizers.py  # optimizers
│   ├── trainer.py  # trainer class
│   └── ...
│
├── data/  # default directory for storing input data
│   └── ...
│
├── datasets/  # contains the dataset definitions
│   ├── dataset1.py
│   └── ...
│
├── logs/  # default directory for storing logs
│   ├── YYYY-MM-DD-HH-MM-SS/  # training logs
│   │   ├── checkpoints/  # model checkpoints
│   │   │   ├── best_checkpoint.pth  # best model checkpoint
│   │   │   └── latest_checkpoint.pth  # latest model checkpoint
│   │   │
│   │   ├── train_losses.json  # train losses
│   │   ├── train_metrics.json  # train metrics
│   │   ├── validation_losses.json  # validation losses
│   │   ├── validation_metrics.json  # validation metrics
│   │   ├── test_loss.json  # test loss
│   │   └── test_metrics.json  # test metrics
│   │
│   └── ...
│
├── models/  # contains the model definitions
│   ├── model1.py
│   └── ...
│
├── tools/  # scripts for training, testing, etc.
│   ├── test.py
│   ├── train.py
│   ├── trainer.py
│   └── ...
│
├── utils/  # utility functions
│   ├── loader.py
│   ├── utils.py
│   └── ...
│
└── requirements.txt  # project dependencies
```

## Configurations

## Future Work

## License

## Acknowledgments
