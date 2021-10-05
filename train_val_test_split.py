"""This module do data splitting for training, testing and validation
procedures.
"""
import argparse as ap
import os

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from utils.splitting import stratify_shuffle_split_subsets

DEFAULT_TRAINING_FRACTIONS = 0.8
COLUMN_NAME_WITH_IMAGES = 'image_name'

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Data splitting.')
    parser.add_argument(
        '--csv_path', type=str, required=True, help='Source .CSV file.'
    )
    parser.add_argument(
        '--path_to_image_folder',
        type=str,
        required=True,
        help='Path to folder with images.'
    )
    parser.add_argument(
        '--splitting_folder_path',
        type=str,
        required=True,
        help='Path to folder in which splitted .csv files will be saved.'
    )
    parser.add_argument(
        '--train_fraction',
        type=float,
        required=False,
        default=DEFAULT_TRAINING_FRACTIONS,
        help='Fraction of data for training subset.'
    )
    args = parser.parse_args()

    if not os.path.exists(args.splitting_folder_path):
        os.mkdir(args.splitting_folder_path)

    df = pd.read_csv(args.csv_path)
    image_names = df[COLUMN_NAME_WITH_IMAGES].str.cat(
        others=['jpg' for _ in range(len(df))],
        sep='.'
    )
    df[COLUMN_NAME_WITH_IMAGES] = image_names.apply(
        lambda image_name: os.path.join(args.path_to_image_folder, image_name)
    )
    df.tags = df.tags.apply(
        lambda x: set(x.split(' '))
    )
    mlb = MultiLabelBinarizer()
    encoded = pd.DataFrame(mlb.fit_transform(df.tags), columns=mlb.classes_)
    encoded_df = pd.concat((df[COLUMN_NAME_WITH_IMAGES], encoded), axis=1)

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(
        encoded_df,
        img_path_column=COLUMN_NAME_WITH_IMAGES,
        train_fraction=args.train_fraction,
        verbose=False
    )
    splits = zip((train_df, valid_df, test_df), ('train', 'valid', 'test'))
    for df, subset in splits:   # noqa: WPS440
        df_name = '.'.join((subset, 'csv'))
        subset_path = os.path.join(args.splitting_folder_path, df_name)
        df.to_csv(subset_path, index=False)
