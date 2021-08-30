"""This module contains DTO for project options."""
import yaml
from pydantic import BaseModel


class PathOptions(BaseModel):
    """Holds paths to files."""

    path_to_images: str
    train_path: str
    val_path: str
    test_path: str


class TrainingOptions(BaseModel):
    """Holds training options."""

    seed: int


class OptionsKeeper(BaseModel):
    """Keeps all project's options."""

    training: TrainingOptions
    path: PathOptions


def get_options(path_to_cfg: str) -> OptionsKeeper:
    """Parse .YAML file with project options and build options object.

    Parameters:
        path_to_cfg: Path to configuration .YAML file.

    Returns:
        Options serialized in object.
    """
    with open(path_to_cfg, 'r') as yf:
        options = OptionsKeeper.parse_obj(yaml.safe_load(yf))
    return options
