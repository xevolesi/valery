"""This module contains utilities for data splitting."""
import logging
import typing as tp

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_stratification


def _split(
    img_urls: np.array,
    labels: np.array,
    sample_distribution_per_fold: tp.Union[None, tp.List[float]] = None,
) -> tp.Tuple[np.array, np.array]:
    stratifier = iterative_stratification.IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=sample_distribution_per_fold
    )

    # This class is a generator that produces k-folds. We just want to iterate
    # it once to make a single static split.
    train_indexes, everything_else_indexes = next(
        stratifier.split(X=img_urls, y=labels)
    )

    num_overlapping_samples = len(
        set(train_indexes).intersection(set(everything_else_indexes))
    )
    if num_overlapping_samples != 0:
        error = (
            'First split failed, {olp} overlapping samples detected'.format(
                olp=num_overlapping_samples
            )
        )
        raise ValueError(error)
    return train_indexes, everything_else_indexes


def _show_split(  # noqa: WPS210
    train_fraction: float,
    y_train: np.array,
    y_dev: np.array,
    y_val: np.array,
    full_dataset: pd.DataFrame,
) -> tp.NoReturn:
    val_test_fraction = (1.0 - train_fraction) / 2
    splits = [
        ('train', train_fraction, y_train),
        ('test', val_test_fraction, y_dev),
        ('val', val_test_fraction, y_val)
    ]

    for subset_name, frac, encodings_collection in splits:
        # Column-wise sum. sum(counts) > n_samples due to imgs with >1 class.
        count_values = np.sum(encodings_collection, axis=0)

        # Skip first col, which is the image key, not a class ID.
        counts = {
            class_id: count_val for class_id, count_val in zip(
                full_dataset.columns[1:], count_values
            )
        }
        message = '{n} subset {f} counts after stratification:{c}'.format(
            n=subset_name, f=round(frac * 100, 2), c=counts
        )
        logging.info(message)


def stratify_shuffle_split_subsets(
    full_dataset: pd.DataFrame,
    img_path_column: str = 'Id',
    train_fraction: float = 0.8,
    verbose: bool = False,
) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified shuffled split a multi-class multi-label dataset into train,
    validation and test subsets.

    Parameters:
        full_dataset: Dataset;
        img_path_column: Name of column that contains paths to images;
        train_fraction: Fraction of samples that go to the train subset;
        verbose: Whether to show log or not.

    Raises:
        ValueError: If dataset contains image duplicates.

    Returns:
        Train, validation and test subsets.
    """
    # Pandas documentation says to use .to_numpy() instead of .values for
    # consistency.
    img_urls = full_dataset[img_path_column].to_numpy()

    # sanity check: no duplicate labels
    if len(img_urls) != len(set(img_urls)):
        raise ValueError('Duplicate image keys detected.')

    labels = full_dataset.drop(
        columns=[img_path_column]
    ).to_numpy().astype(int)

    # NOTE generators are replicated across workers. Do stratified shuffle
    # split beforehand.
    logging.info('Stratifying dataset iteratively. this may take a while.')

    # NOTE: splits >2 broken;
    # https://github.com/scikit-multilearn/scikit-multilearn/issues/209
    # so, do 2 rounds of iterative splitting.
    train_indexes, everything_else_indexes = _split(
        img_urls, labels, [1.0 - train_fraction, train_fraction]
    )
    x_train, x_else = (
        img_urls[train_indexes], img_urls[everything_else_indexes]
    )
    y_train, y_else = (
        labels[train_indexes, :], labels[everything_else_indexes, :]
    )

    dev_indexes, val_indexes = _split(x_else, y_else)
    x_dev, x_val = x_else[dev_indexes], x_else[val_indexes]
    y_dev, y_val = y_else[dev_indexes, :], y_else[val_indexes, :]

    if verbose:
        _show_split(train_fraction, y_train, y_dev, y_val, full_dataset)

    # Combine (x,y) data into dataframes.
    train_subset = pd.DataFrame(y_train)
    train_subset.insert(0, img_path_column, pd.Series(x_train))
    train_subset.columns = full_dataset.columns

    dev_subset = pd.DataFrame(y_dev)
    dev_subset.insert(0, img_path_column, pd.Series(x_dev))
    dev_subset.columns = full_dataset.columns

    val_subset = pd.DataFrame(y_val)
    val_subset.insert(0, img_path_column, pd.Series(x_val))
    val_subset.columns = full_dataset.columns

    logging.info('Stratifying dataset is completed.')

    return train_subset, val_subset, dev_subset
