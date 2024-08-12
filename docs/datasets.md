# Datasets
This page contains documentation about the management of datasets in the project.

## Table Of Contents
- [Creating A Dataset](#creating-a-dataset)
- [Using A Dataset](#using-a-dataset)
- [Observations](#observations)

## Creating A Dataset
To create a new dataset, follow these steps:

1. Create a new file in the `src/datasets` directory. This fill will contain the dataset implementation.
2. Implement the dataset class. The class should inherit from `torch.utils.data.Dataset`.
3. Make sure that the dataset contains the following methods:
    - `__init__(self, split, ...)` to initialize the dataset. The possible values for the `split` parameter are `train`, `val` and `test`.
    - `__len__(self)` to return the number of samples in the dataset.
    - `__getitem__(self, idx)` to return a tuple `(sample, target)` where `sample` is the input data and `target` is the target label.
4. Add the dataset to the `classes` list inside the `datasets/__init__.py` file.

## Using A Dataset
To use a dataset in the project, follow these steps:

1. Add the dataset configuration to the `config/datasets.yaml` file.
2. Use the dataset reference in the intended experiment configuration file.

For more information about the configuration files, check the [Configuration Files](configs.md) page.

## Observations
Whenever using randomness in the dataset, make sure to set the random seed to ensure reproducibility. The random seed is defined in the `utils/constants.py` file as `Constants.Miscellaneous.SEED`. The following code snippet shows an example of how to set the random seed in the dataset:

```python
generator = torch.Generator().manual_seed(Constants.Miscellaneous.SEED)
splitted_data = random_split(dataset, [num_labeled, len(dataset) - num_labeled], generator=generator)
```
