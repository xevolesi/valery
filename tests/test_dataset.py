import torch
import numpy as np
import pandas as pd
import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2

from dataset import PlanetDataset
from utils.options import get_options


options = get_options('config.yml')


def test_dataset_torch():
    df = pd.read_csv(options.tests.csv_path)
    image_size = (options.training.image_size, options.training.image_size)
    dataset = PlanetDataset(
        dataframe=df,
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
    df = pd.read_csv(options.tests.csv_path)
    dataset = PlanetDataset(dataframe=df)
    for data_point in dataset:
        image, label = data_point
        assert isinstance(image, np.ndarray)
        assert isinstance(label, np.ndarray)
        assert tuple(label.shape) == (17,)
