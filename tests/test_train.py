"""Tests for training procedure"""
from unittest.mock import Mock

from utils.config import get_config
from utils.training import train_and_validate

config = get_config('config.yml')
config.path.train_path = config.tests.csv_path
config.path.val_path = config.tests.csv_path
config.training.epochs = 1
config.training.device = 'cpu'


def test_train():
    """Test training procedure."""
    train_and_validate(config, Mock())
