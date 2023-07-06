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
│   └── ...
│
├── data/  # default directory for storing input data
│   ├── data1.csv
│   └── ...
│
├── datasets/  # contains the dataset definitions
│   ├── dataset1.py
│   └── ...
│
├── logs/  # default directory for storing logs
│   ├── train-YYYY-MM-DD-HH-MM-SS/  # training logs
│   │   ├── checkpoints/  # model checkpoints
│   │   │   ├── checkpoint1.pth
│   │   │   └── ...
│   │   ├── train_losses.log  # train losses
│   │   ├── train_metrics.log  # train metrics
│   │   ├── validation_losses.log  # validation losses
│   │   ├── validation_metrics.log  # validation metrics
│   │   └── ...
│   │
│   ├── test-YYYY-MM-DD-HH-MM-SS/  # testing logs
│   │   ├── test_losses.log  # test losses
│   │   ├── test_metrics.log  # test metrics
│   │   └── ...
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
│   ├── utils.py
│   └── ...
│
└── requirements.txt  # project dependencies
```

## Configurations

## Future Work

## License

## Acknowledgments
