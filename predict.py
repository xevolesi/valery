"""This module shows the example of inference for trained model."""
import argparse as ap

import cv2
import numpy as np
import torch

from utils.config import get_config
from utils.general import preprocess_imagenet

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Do prediction on single image.')
    parser.add_argument(
        '--model_path', type=str, help='Path to model file.', required=True
    )
    parser.add_argument(
        '--image_path', type=str, help='Path to image file.', required=True
    )
    args = parser.parse_args()
    config = get_config('config.yml')
    model = torch.jit.load(args.model_path)
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_imagenet(image, config.training.image_size)
    probas = model(torch.Tensor(image)).detach().numpy()
    indices = (probas > model.thresholds)[0]
    classes = np.array(model.classes)[indices].tolist()
    print(  # noqa: WPS421
        'Classes for this image: {cls_list}.'.format(
            cls_list=', '.join(classes)
        ),
    )
