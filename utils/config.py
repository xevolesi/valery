"""This module contains DTO for project options."""
import typing as tp

import yaml
from pydantic import BaseModel, create_model, validator


class ClearMLConfig(BaseModel):
    """Holds ClearML options."""

    project_name: str
    experiment_number: int


class TestsConfig(BaseModel):
    """Holds options for tests."""

    csv_path: str


class PathConfig(BaseModel):
    """Holds paths to files."""

    path_to_images: str
    train_path: str
    val_path: str
    test_path: str
    weights_folder_path: str


class TrainingConfig(BaseModel):
    """Holds training options."""

    seed: int
    epochs: int
    device: str
    num_workers: int
    image_size: int
    batch_size: int


class ModelConfig(BaseModel):
    """Holds model's options."""

    name: str
    n_classes: int
    pretrained: bool


class Config(BaseModel):
    """Keeps all project's options."""

    training: TrainingConfig
    path: PathConfig
    tests: TestsConfig
    model: ModelConfig
    clearml: ClearMLConfig
    optimizer: tp.Any
    scheduler: tp.Any
    criterion: tp.Any
    augmentations: tp.Any

    @validator('optimizer', 'scheduler', 'criterion', 'augmentations')
    def build(
        cls,                    # noqa: N805
        model_parameters,
        values,                 # noqa: WPS110
        config,
        field
    ) -> BaseModel:
        """
        Build pydantic model for configuration.

        Parameters:
            model_parameters: Dictionary with parameters for builded model;
            values: Already builded fields for current model;
            config: Build configuration;
            field: Current field configuration.

        Returns:
            Pydantic model.
        """
        model_name = ''.join((field.name.capitalize(), 'Config'))
        return create_model(model_name, **model_parameters)()


def get_config(path_to_cfg: str) -> Config:
    """Parse .YAML file with project options and build options object.

    Parameters:
        path_to_cfg: Path to configuration .YAML file.

    Returns:
        Options serialized in object.
    """
    with open(path_to_cfg, 'r') as yf:
        yml_file = yaml.safe_load(yf)
        config = Config.parse_obj(yml_file)
    return config
