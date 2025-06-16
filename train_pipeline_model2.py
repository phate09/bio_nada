"""
Given a folder with name data and the corresponding csv file label.csv, this script trains a neural network
The data is split in 5 folds and the accuracy and f1 score are averaged over the folds.
The network tries to predict the cell label of each cell.
The only values allowed are 0, 1 and 2 in the cell label column.
"""
import os

import pandas as pd

from pre_process import get_dataframe, \
    preprocess_cell_label
from training.methods import train_cell_model, generate_run_folder
from pathlib import Path

if __name__ == '__main__':
    (pd.set_option('display.expand_frame_repr', False))
    config = {"sampler": "None", # ROS or RUS or None. If None, no sampling is used.
              "loss": "CE", # CE or Focal
              "n_splits": 2,
              "lr": 1e-3,
              "lr_decay_factor": 0.99,
              "batch_size": 2 ** 14,
              "epochs": 50,
              "alpha": 0.5, # for focal loss
              "gamma": 2, # for focal loss
              "seed": 0,
              "label_column": "cell_label2",  # which column to use to predict the result.
              "kfold_group_column": "id_random",
              # which column to use for stratified k-folds. "id_random" is used for keeping the patients together.
              "feature_columns": ["FSC", "SSC", "CD16 AF488", "CD14-PE"]
              }

    print("Preparing dataframe")
    master_df = get_dataframe()
    master_df = preprocess_cell_label(master_df)
    master_df = master_df[master_df['cell_label'] == 1] # filter only cells with label 1
    master_df.reset_index(drop=True, inplace=True)

    os.chdir(generate_run_folder(prefix='model2'))  # change working directory to the run folder
    train_cell_model(config=config, master_df=master_df)
