"""This module contains some general utilities."""
import os
import pydoc
import random
import typing as tp

import cv2
import numpy as np
import torch
from pydantic import BaseModel

from utils.config import Config

MAX_PIXEL_INTENCITY = 255


def object_from_pydantic(
    pydantic_model: BaseModel,
    parent: tp.Optional[BaseModel] = None,
    **additional_kwargs: tp.Dict[str, tp.Union[float, str, int]],
) -> tp.Any:
    """
    Parse pydantic model and build instance of provided type.

    Parameters:
        pydantic_model: Pydantic model;
        parent: Parent model;
        additional_kwargs: Additional arguments for instantiation procedure.

    Returns:
        Intance of provided type.
    """
    kwargs = pydantic_model.dict().copy()
    object_type = kwargs.pop('algo')
    for param_name, param_value in additional_kwargs.items():
        kwargs.setdefault(param_name, param_value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)

    return pydoc.locate(object_type)(**kwargs)


def seed_everything(config: Config) -> None:
    """
    One function to seed 'em all!

    Parameters:
        config: Project's option object.
    """
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    os.environ['PYTHONHASHSEED'] = str(config.training.seed)
    torch.cuda.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id) -> None:
    """
    Fix seed for current worker. This should fix the bug with identical augs.

    Parameters:
        worker_id: ID of current dataloader worker.
    """
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_cpu_state_dict(model: torch.nn.Module) -> tp.Dict:
    """
    Return model's state dict located on CPU device.

    Parameters:
        model: Your cool neural network.

    Returns:
        State dictionary with CPU tensors.
    """
    return {k: v.cpu() for k, v in model.state_dict().items()}


def preprocess_imagenet(image: np.ndarray, image_size: int) -> np.ndarray:
    """
    Do standard ImageNet preprocessing procedure.

    Parameters:
        image: Source image;
        image_size: Target image size. Image will be resized to this size.

    Returns:
        Preprocessed image tensor.
    """
    im = image.astype(np.float32)
    im /= MAX_PIXEL_INTENCITY
    im = cv2.resize(im, (image_size, image_size))
    im = np.transpose(im, (2, 0, 1))
    im -= np.array([0.485, 0.456, 0.406])[:, None, None]
    im /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return np.expand_dims(im, axis=0)
