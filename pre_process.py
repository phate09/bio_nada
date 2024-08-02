import os

import numpy as np
import pandas as pd
import pathlib
import csv

import sklearn.metrics
import torch
from torch.utils.data import DataLoader, TensorDataset

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
                      nn.Linear(64, 1),  # just 1 output because of 2 classes
                      nn.Sigmoid()  # just sigmoid instead of softmax
                      )
criterion = nn.BCELoss()  # Binary Cross Entropy
optimiser = optim.Adam(model.parameters(), lr=1e-4)
# %% ---- dataloaders
batch_size = 1024
train_loader = DataLoader(TensorDataset(x_tensor_train, y_tensor_train), batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(TensorDataset(x_tensor_test, y_tensor_test), batch_size=batch_size,
                         shuffle=False)

# %% ------- training
n_epochs = 1
train_accs = []
train_losses = []
for epoch in range(n_epochs):
    correct_train = 0
    total_train = 0
    train_loss = 0
    for X_batch, y_batch in train_loader:
        y_batch = y_batch.float().reshape(-1, 1)  # reshape and cast to float
        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        assert y_pred.shape == y_batch.shape
        train_loss += loss.item()
        # Calculate training accuracy
        total_train += y_batch.size(0)  # size of the batch
        correct_train += (y_pred.round() == y_batch).sum().item()  # number of correct items

        # Backward pass and optimization
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    # ---- validation step
    print(f"Training accuracy = {correct_train / total_train:.2f}")
    model.eval()
    y_pred = model(x_tensor_test)
    validation_accuracy = (y_pred.round() == y_tensor_test.float().reshape(-1, 1)).float().mean()
    print(f"Validation accuracy = {validation_accuracy:.2f}")
