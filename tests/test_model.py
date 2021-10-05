"""Tests for DL model."""
import torch

from models.planet_classifier import PlanetClassifier
from utils.config import get_config

config = get_config('config.yml')


def test_model():
    """Test model's output shape."""
    model = PlanetClassifier(**config.model.dict())
    image_size = (config.training.image_size, config.training.image_size)
    with torch.no_grad():
        logits = model(torch.randn((5, 3, *image_size)))
    assert tuple(logits.shape) == (5, 17)
