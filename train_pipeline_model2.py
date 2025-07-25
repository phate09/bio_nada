"""
Given a folder with name data and the corresponding csv file label.csv, this script trains a neural network
The data is split in K folds and the accuracy and f1 score are averaged over the folds.
The network tries to predict the cell label 2 of each cell. That is the type of cells where the primary label is 1.
The values in the cell label column are from 0 to 4 but they are rescaled to -1 to 3.
Furthermore, the cell label 2 value of 3 is removed as requested.
"""
import os
import pickle
from pathlib import Path

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from pre_process import get_dataframe, \
    preprocess_cell_label
from training.methods import train_cell_model, generate_run_folder
from pathlib import Path

if __name__ == '__main__':
    (pd.set_option('display.expand_frame_repr', False))
    config = {"sampler": "None",  # ROS or RUS or None. If None, no sampling is used.
              "loss": "Focal",  # CE or Focal
              "n_splits": 2,
              "lr": 1e-3,
              "lr_decay_factor": 0.8,
              "batch_size": 2 ** 18,
              "epochs": 5,
              "alpha": [0.45, 0.35, 0.20],  # for focal loss
              "gamma": 1.5,  # for focal loss
              "seed": 0,
              "label_column": "cell_label2",  # which column to use to predict the result.
              "kfold_group_column": None,  # "id_random",
              # which column to use for stratified k-folds. "id_random" is used for keeping the patients together.
              "feature_columns": ["FSC", "SSC", "CD16 AF488", "CD14-PE"]
              }

    cache_file = Path(".preprocessed_model2")
    if cache_file.exists():
        print("Loading preprocessed data from cache")
        with open(cache_file, 'rb') as f:
            master_df = pickle.load(f)
    else:
        print("Preparing dataframe")
        master_df = get_dataframe(label_file="lab-15.csv")
        master_df = preprocess_cell_label(master_df)
        master_df = master_df[master_df['cell_label'] == 1]  # filter only cells with label 1
        master_df = master_df[master_df['cell_label2'] < 3]  # filter three output classes 0, 1, 2
        master_df.reset_index(drop=True, inplace=True)

        print("Saving preprocessed data to cache")
        with open(cache_file, 'wb') as f:
            pickle.dump(master_df, f)

    run_folder = generate_run_folder(prefix='model2')
    os.chdir(run_folder)  # change working directory to the run folder
    train_cell_model(config=config, master_df=master_df,run_folder=run_folder)
