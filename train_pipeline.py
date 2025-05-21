"""
Given a folder with name data and the corresponding csv file label.csv, this script trains a neural network
The data is split in 5 folds and the accuracy and f1 score are averaged over the folds.
The network tries to predict the cell label of each cell.
The only values allowed are 0, 1 and 2 in the cell label column.
"""
import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim

from neural_network import get_simple_model, neural_network_2, neural_network_3
from pre_process import create_dataloaders, create_tensor_from_df, get_dataframe, \
    preprocess_cell_label
import progressbar
import pandas as pd

(pd.set_option('display.expand_frame_repr', False))
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Preparing dataframe")
master_df = get_dataframe()
master_df = preprocess_cell_label(master_df)
# Discard unused columns
master_df.drop(['cell_label2',"Label","id_random"], axis=1, inplace=True) # only cell_label remains
print(f"Start. Using {device}")
n_features = len([col for col in master_df.columns if col not in ['cell_label','id_random','Label','cell_label2']])
output_dim = master_df['cell_label'].nunique()
model = neural_network_3(n_features,output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=1e-4)
k_fold = StratifiedKFold(n_splits=5, shuffle=True)

accuracy_list = []
f1_score_list = []
precision_list = []
recall_list = []
for i, (train_idx, test_idx) in enumerate(k_fold.split(master_df, master_df.iloc[:, -1])):  # k-fold
    tensors = create_tensor_from_df(master_df.loc[train_idx], master_df.loc[test_idx])
    # use the * to reuse the individual variables in input
    train_loader, eval_loader = create_dataloaders(*tensors, batch_size=16384)
    n_epochs = 3
    train_accs = []
    train_losses = []
    for epoch in range(n_epochs):
        # ---- training step
        correct = 0
        n_examples = 0
        train_loss = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for X_batch, y_batch in progressbar.progressbar(train_loader, prefix="Batch"):
            # Forward pass
            X_batch = X_batch.to(device)
            y_batch = y_batch.squeeze().to(device).long()
            optimiser.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            train_loss += loss.item()
            # Calculate metrics
            n_examples += y_batch.size(0)  # size of the batch
            y_pred = logits.argmax(dim=1)
            correct += (y_pred.round() == y_batch).sum().item()  # number of correct items
            true_positive += ((y_pred.round() == y_batch) & (y_batch == 1)).sum().item()
            true_negative += ((y_pred.round() == y_batch) & (y_batch == 0)).sum().item()
            false_positive += ((y_pred.round() != y_batch) & (y_batch == 1)).sum().item()
            false_negative += ((y_pred.round() != y_batch) & (y_batch == 0)).sum().item()
            # Backward pass and optimization
            loss.backward()
            optimiser.step()

        print(f"Fold {i + 1} Epoch {epoch + 1}")  # +1 because i starts from 0
        precision = true_positive / max((true_positive + false_positive), 1)  # prevents div by 0
        recall = true_positive / max((true_positive + false_negative), 1)  # prevents div by 0
        f1_score = 2 * (precision * recall) / max((precision + recall), 1)  # prevents div by 0
        accuracy = correct / n_examples
        print(f"TRAINING: accuracy={accuracy:.2f}, f1_score={f1_score:.2f}, "
              f"precision={precision:.2f}, recall={recall:.2f}")
        # ---- validation step
        model.eval()  # put the model in evaluation mode
        correct = 0
        n_examples = 0
        train_loss = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for X_batch, y_batch in eval_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.squeeze().to(device).long()
            # Forward pass
            y_pred = model(X_batch).argmax(dim=1)
            # Calculate training accuracy
            n_examples += y_batch.size(0)  # size of the batch
            correct += (y_pred.round() == y_batch).sum().item()  # number of correct items
            true_positive += ((y_pred.round() == y_batch) & (y_batch == 1)).sum().item()
            true_negative += ((y_pred.round() == y_batch) & (y_batch == 0)).sum().item()
            false_positive += ((y_pred.round() != y_batch) & (y_batch == 1)).sum().item()
            false_negative += ((y_pred.round() != y_batch) & (y_batch == 0)).sum().item()
        precision = true_positive / max((true_positive + false_positive), 1)  # prevents div by 0
        recall = true_positive / max((true_positive + false_negative), 1)  # prevents div by 0
        f1_score = 2 * (precision * recall) / max((precision + recall), 1)  # prevents div by 0
        accuracy = correct / n_examples
        print(f"TEST: accuracy={accuracy:.2f}, f1_score={f1_score:.2f}, "
              f"precision={precision:.2f}, recall={recall:.2f}")
    accuracy_list.append(accuracy)
    f1_score_list.append(f1_score)
    precision_list.append(precision)
    recall_list.append(recall)
mean_accuracy = np.mean(accuracy_list)
mean_f1 = np.mean(f1_score_list)
mean_precision = np.mean(precision_list)
mean_recall = np.mean(recall_list)
print(f"FINAL: accuracy={mean_accuracy:.2f}, f1_score={mean_f1:.2f}, "
      f"precision={mean_precision:.2f}, recall={mean_recall:.2f}")
