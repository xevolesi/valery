"""Utilities for training procedure."""
import typing as tp

import numpy as np
import pandas as pd
import torch
from clearml import Logger
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.planet_dataset import PlanetDataset
from models.planet_classifier import PlanetClassifier
from utils import general
from utils.augmentations import get_augmentations
from utils.config import Config
from utils.metrics import f2_score


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.BCEWithLogitsLoss,
    device: torch.device,
    current_epoch: int
) -> tp.Tuple[float, float]:
    """
    Perform validation procedure.

    Parameters:
        model: Your cool neural network;
        dataloader: Training data loader;
        criterion: Loss function;
        device: Device on which perform training;
        current_epoch: The number of current epoch.

    Returns:
        Validation loss and validation metric values.
    """
    model.eval()
    epoch_loss = 0
    progress = tqdm(
        dataloader,
        total=len(dataloader),
        desc='Val epoch #{current_epoch}'.format(current_epoch=current_epoch)
    )
    score = 0
    with torch.no_grad():
        for i, batch in enumerate(progress):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            epoch_loss += loss.item()
            score += f2_score(torch.sigmoid(logits), labels).item()
            progress.set_postfix_str(
                s='Loss: {eloss:.5f}, F2: {f2score:.5f}'.format(
                    eloss=epoch_loss / (i + 1), f2score=score / (i + 1)
                )
            )
    return epoch_loss / len(dataloader), score / len(dataloader)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    current_epoch: int
) -> float:
    """
    Perform training procedure on one epoch.

    Parameters:
        model: Your cool neural network;
        dataloader: Training data loader;
        criterion: Loss functions;
        optimizer: Optimization algorithm;
        scaler: Gradinet scaler for mixed precision;
        device: Device on which perform training;
        current_epoch: The number of current epoch.

    Returns:
        Training loss value on current epoch.
    """
    model.train()
    epoch_loss = 0
    progress = tqdm(
        dataloader,
        total=len(dataloader),
        desc='Train epoch #{current_epoch}'.format(current_epoch=current_epoch)
    )
    for i, batch in enumerate(progress):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        optimizer.zero_grad()
        if device.type == 'cuda':
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        progress.set_postfix_str(
            s='Loss: {eloss:.5f}'.format(eloss=epoch_loss / (i + 1))
        )
    return epoch_loss / len(dataloader)


def train_and_validate(config: Config, logger: Logger) -> tp.Dict:
    """
    Perform training and validation procedure.

    Parameters:
        config: Project's config object;
        logger: ClearML logger.

    Returns:
        Best checkpoint with best score.
    """
    device = torch.device(config.training.device)
    train_df = pd.read_csv(config.path.train_path)
    valid_df = pd.read_csv(config.path.val_path)
    train_augs, valid_augs = get_augmentations(config)
    train_dataset = PlanetDataset(train_df, train_augs)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=general.worker_init_fn,
        num_workers=config.training.num_workers
    )
    valid_dataset = PlanetDataset(valid_df, valid_augs)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=2 * config.training.batch_size,
        pin_memory=True,
        worker_init_fn=general.worker_init_fn,
        num_workers=config.training.num_workers,
    )
    model = PlanetClassifier(**config.model.dict()).float().to(device)
    optimizer = general.object_from_pydantic(
        config.optimizer, params=model.parameters()
    )
    scheduler = general.object_from_pydantic(
        config.scheduler, optimizer=optimizer
    )
    scaler = GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()
    best_score = 0
    best_state_dict = None
    for epoch in range(config.training.epochs):
        np.random.seed(config.training.seed + epoch)
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch + 1
        )
        valid_loss, valid_score = validate(
            model, valid_dataloader, criterion, device, epoch + 1
        )
        logger.report_scalar('Losses', 'Train loss', train_loss, epoch)
        logger.report_scalar('Losses', 'Validation loss', valid_loss, epoch)
        logger.report_scalar('Metrics', 'F2-score', valid_score, epoch)
        logger.report_scalar(
            'LR', 'Learning rates', scheduler.get_last_lr()[0], epoch
        )
        scheduler.step()

        if valid_score >= best_score:
            best_score = valid_score
            best_state_dict = general.get_cpu_state_dict(model)
    return best_state_dict
