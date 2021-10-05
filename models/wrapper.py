"""This module contains wrapper for TorchScript."""
import typing as tp

import torch

from models.planet_classifier import PlanetClassifier


class ModelWrapper(torch.nn.Module):
    """Model's wrapper for scripting."""

    def __init__(
        self,
        classifier: PlanetClassifier,
        classes: tp.Tuple[str, ...],
        size: tp.Tuple[int, int],
        thresholds: tp.Tuple[float, ...]
    ) -> None:
        """
        Parameters:
            classifier: Classification model;
            classes: The names of target classes;
            size: Image size for inference;
            thresholds: Thresholds for probabilities for each class.
        """
        super().__init__()
        self.model = classifier
        self.classes = classes
        self.size = size
        self.thresholds = thresholds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input tensor to probabilitites.

        Parameters:
            x: Tensorized image.

        Returns:
            Probabilities for each class.
        """
        return torch.sigmoid(self.model.forward(x))
