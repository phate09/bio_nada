import os

import numpy as np
import pandas as pd
import pathlib
import csv

import torch

# %% ---- merge all data csv in dataframe

data_folder = "data"
master_df = pd.DataFrame()
list_df = []
for filee in os.listdir(
    data_folder):  # Loop through CSV files in the dynamically specified directory
    if filee.startswith(".") or not filee.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(data_folder, filee))  # Read the CSV data into a DataFrame
    df["id_random"] = int(filee.removesuffix(".csv"))
    list_df.append(df)
master_df = pd.concat(list_df)
print(master_df)

# %%----- append label column to dataframe
label_file = "label.csv"
label_df = pd.read_csv(label_file)

master_df = master_df.merge(label_df, left_on="id_random", right_on="id_random", how="left")
# %% ----- train/test set
from sklearn.model_selection import train_test_split

train_set_df, test_set_df = train_test_split(master_df, train_size=0.8, random_state=0)
# %% ------ create tensor
x_tensor_train = torch.tensor(train_set_df.iloc[:, :-2].values,
                              dtype=torch.float)  # exclude last two columns
y_tensor_train = torch.tensor(train_set_df.iloc[:, -1].values, dtype=torch.bool)
assert x_tensor_train.shape[0] == y_tensor_train.shape[0]
x_tensor_test = torch.tensor(test_set_df.iloc[:, :-2].values,
                             dtype=torch.float)  # exclude last two columns
y_tensor_test = torch.tensor(test_set_df.iloc[:, -1].values, dtype=torch.bool)
assert x_tensor_test.shape[0] == y_tensor_test.shape[0]
# %% ------ define network and loss
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(6, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 2),
                      nn.Softmax(dim=1))
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)
# %% ---- dataloaders
