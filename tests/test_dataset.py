"""Tests for custom dataset class."""
import albumentations as album
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2

from dataset.planet_dataset import PlanetDataset
from utils.options import get_options

options = get_options('config.yml')


def test_dataset_torch():
    """Test dataset class with torch tensors as the outputs."""
    image_size = (options.training.image_size, options.training.image_size)
    dataset = PlanetDataset(
        dataframe=pd.read_csv(options.tests.csv_path),
        transform=album.Compose(
            [
                album.Resize(*image_size),
                album.Normalize(),
                ToTensorV2()
            ]
        )
    )
    for data_point in dataset:
        image, label = data_point
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert tuple(image.shape) == (3, *image_size)
        assert tuple(label.shape) == (17,)


def test_dataset_numpy():
    """Test dataset class with numpy arrays as the outputs."""
    dataset = PlanetDataset(dataframe=pd.read_csv(options.tests.csv_path))
    for data_point in dataset:
        image, label = data_point
        assert isinstance(image, np.ndarray)
        assert isinstance(label, np.ndarray)
        assert tuple(label.shape) == (17,)
