"""Tests for custom dataset class."""
import typing as tp

import numpy as np
import pandas as pd
import torch

from dataset.planet_dataset import PlanetDataset
from utils.augmentations import get_augmentations
from utils.config import get_config

config = get_config('config.yml')


def _iterate_over_dataset(
    dataset: PlanetDataset,
    assertion_data_type: tp.Union[torch.Tensor, np.ndarray] = torch.Tensor
) -> None:
    """
    Iterate over provided dataset.

    Parameters:
        dataset: Dataset for iteration;
        assertion_data_type: Which type should the data be;
    """
    image_size = (config.training.image_size, config.training.image_size)
    for data_point in dataset:
        image, label = data_point
        assert isinstance(image, assertion_data_type)
        assert isinstance(label, assertion_data_type)
        if assertion_data_type is torch.Tensor:
            assert tuple(image.shape) == (3, *image_size)
        assert tuple(label.shape) == (17,)


def test_dataset_torch():
    """Test dataset class with torch tensors as the outputs."""
    train_augs, valid_augs = get_augmentations(config)
    dataset = PlanetDataset(
        dataframe=pd.read_csv(config.tests.csv_path),
        transform=train_augs
    )
    _iterate_over_dataset(dataset)
    dataset = PlanetDataset(
        dataframe=pd.read_csv(config.tests.csv_path),
        transform=valid_augs
    )
    _iterate_over_dataset(dataset)


def test_dataset_numpy():
    """Test dataset class with numpy arrays as the outputs."""
    dataset = PlanetDataset(dataframe=pd.read_csv(config.tests.csv_path))
    _iterate_over_dataset(dataset, assertion_data_type=np.ndarray)
