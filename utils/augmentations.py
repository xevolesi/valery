"""This module contains different augmentations methods."""
import typing as tp

import albumentations as album
from albumentations.core import serialization

from utils.config import Config


def get_augmentations(
    config: Config
) -> tp.Tuple[album.Compose, album.Compose]:
    """
    Build augmentation pipelines for training and validation.

    Parameters:
        config: Project's config object.

    Returns:
        Augmentation pipelines for training and validation.
    """
    train_augs = serialization.from_dict(config.augmentations.train)
    validation_augs = serialization.from_dict(config.augmentations.val)
    return train_augs, validation_augs
