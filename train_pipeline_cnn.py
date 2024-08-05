import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim

from neural_network import get_simple_model, neural_network_2, conv_neural_network
from pre_process import create_dataloaders, create_tensor_from_df, get_dataframe
import progressbar
import pandas as pd

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Start. Using {device}")
model = conv_neural_network().to(device)
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=1e-3)
k_fold = StratifiedKFold(n_splits=5, shuffle=True)
print("Preparing dataframe")
master_df = get_dataframe().groupby("id_random")
labels_df = pd.read_csv("label.csv")
accuracy_list = []
f1_score_list = []
precision_list = []
recall_list = []
for i, (train_idx, test_idx) in enumerate(k_fold.split(labels_df, labels_df.iloc[:, -1])):  # k-fold
    train_groups = labels_df.loc[train_idx]
    test_groups = labels_df.loc[test_idx]
    train_groups_names = train_groups.iloc[:, 0]
    n_epochs = 30
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
        for group in progressbar.progressbar(train_groups_names,prefix="Group"):
            df = master_df.get_group(group)[:250000]
            x_tensor_train = torch.tensor(df.iloc[:, :-2].values,
                                          dtype=torch.float).to(device).t()  # exclude last two columns
            y_tensor_train = torch.tensor(df.iloc[0, -1], dtype=torch.float).to(device) # we only care about the first element of the group
            y_pred = model(x_tensor_train)
            y_pred = y_pred.squeeze()
            loss = criterion(y_pred, y_tensor_train)
            y_batch = y_tensor_train
            n_examples += 1  # size of the batch
            correct += (y_pred.round() == y_batch).sum().item()  # number of correct items
            true_positive += ((y_pred.round() == y_batch) & (y_batch == 1)).sum().item()
            true_negative += ((y_pred.round() == y_batch) & (y_batch == 0)).sum().item()
            false_positive += ((y_pred.round() != y_batch) & (y_batch == 1)).sum().item()
            false_negative += ((y_pred.round() != y_batch) & (y_batch == 0)).sum().item()
            # Backward pass and optimization
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f"Fold {i + 1} Epoch {epoch + 1}")  # +1 because i starts from 0
        precision = true_positive / max((true_positive + false_positive), 1)  # prevents div by 0
        recall = true_positive / max((true_positive + false_negative), 1)  # prevents div by 0
        f1_score = 2 * (precision * recall) / max((precision + recall), 1)  # prevents div by 0
        accuracy = correct / n_examples
        print(f"TRAINING: accuracy={accuracy:.2f}, f1_score={f1_score:.2f}, "
              f"precision={precision:.2f}, recall={recall:.2f}")