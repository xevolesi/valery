"""This module contains dataset class definition for Planet project."""
import typing as tp

import albumentations as album
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DataPoint = tp.Tuple[
    tp.Union[torch.Tensor, np.ndarray],
    tp.Union[torch.Tensor, np.ndarray]
]


class PlanetDataset(Dataset):
    """Dataset for model deep learning things."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: tp.Optional[album.Compose] = None
    ) -> None:
        """
        Parameters:
            dataframe: Dataframe with paths and labels;
            transform: Augmentation transformations.
        """
        self.images: tp.Tuple[str] = tuple(dataframe['image_name'])
        self.labels: np.ndarray = dataframe.iloc[:, 1:].values
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.list_images)

    def __getitem__(self, index: int) -> DataPoint:
        """
        Pick one data point.

        Parameters:
            index: The index of data point.

        Returns:
            Image and corresponding label.
        """
        image_path = self.images[index]
        labels = self.labels[index, :]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
            labels = torch.Tensor(labels)
        return image, labels
