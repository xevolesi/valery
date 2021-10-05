"""Main module. Run NN training procedure."""
import os

import torch
from clearml import Task

from models.planet_classifier import PlanetClassifier
from models.wrapper import ModelWrapper
from utils.config import get_config
from utils.general import seed_everything
from utils.training import train_and_validate

CLASS_NAMES = (
    'Agriculture',
    'Artisinal_mine',
    'Bare ground',
    'Blooming',
    'Blow down',
    'Clear',
    'Cloudy',
    'Conventional mine',
    'Cultivation',
    'Habitation',
    'Haze',
    'Partly cloudy',
    'Primary',
    'Road',
    'Selective logging',
    'Slash burn',
    'Water'
)


if __name__ == '__main__':
    config = get_config('config.yml')
    seed_everything(config)
    task = Task.init(
        project_name=config.clearml.project_name,
        task_name='{project}-experiment {exp_num}'.format(
            project=config.clearml.project_name,
            exp_num=str(config.clearml.experiment_number)
        )
    )
    task.connect(config.dict())
    logger = task.get_logger()
    best_checkpoint = train_and_validate(config, logger)
    model = PlanetClassifier(**config.model.dict())
    model.load_state_dict(best_checkpoint)
    wrapper = ModelWrapper(
        classifier=model,
        classes=CLASS_NAMES,
        size=(config.training.image_size, config.training.image_size),
        thresholds=tuple([0.5 for _ in range(len(CLASS_NAMES))])
    )
    scripted = torch.jit.script(wrapper)
    scripted.save(
        os.path.join(config.path.weights_folder_path, 'scripted_model.pt')
    )
