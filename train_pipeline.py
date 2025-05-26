"""
Given a folder with name data and the corresponding csv file label.csv, this script trains a neural network
The data is split in 5 folds and the accuracy and f1 score are averaged over the folds.
The network tries to predict the cell label of each cell.
The only values allowed are 0, 1 and 2 in the cell label column.
"""
import random

import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR

from neural_network import get_simple_model, neural_network_2, neural_network_3
from pre_process import create_dataloaders, create_tensor_from_df, get_dataframe, \
    preprocess_cell_label
import progressbar
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, \
    f1_score, accuracy_score,classification_report
import matplotlib.pyplot as plt


(pd.set_option('display.expand_frame_repr', False))
seed = 0
config = {"sampler": "ROS",}
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Preparing dataframe")
master_df = get_dataframe()
master_df = preprocess_cell_label(master_df)
# Discard unused columns
master_df.drop(['cell_label2', "Label", "id_random"], axis=1,
               inplace=True)  # only cell_label remains
print(f"Start. Using {device}")
n_features = len([col for col in master_df.columns if
                  col not in ['cell_label', 'id_random', 'Label', 'cell_label2']])
output_dim = master_df['cell_label'].nunique()

k_fold = StratifiedKFold(n_splits=5, shuffle=True)
if config["sampler"] == "RUS":
    rus = RandomUnderSampler(random_state=seed, replacement=False)
elif config["sampler"] == "ROS":
    rus = RandomOverSampler(random_state=seed)
else:
    rus = None

accuracy_list = []
f1_score_list = []
precision_list = []
recall_list = []
for i, (train_idx, test_idx) in enumerate(k_fold.split(master_df, master_df.iloc[:, -1])):  # k-fold
    train_groups = master_df.loc[train_idx]
    test_groups = master_df.loc[test_idx]
    if rus is not None:
        X_resampled, y_resampled = rus.fit_resample(train_groups,
                                                    train_groups.iloc[:, -1].values)
    else:
        X_resampled = train_groups
    train_stats_master_df = X_resampled
    eval_stats_master_df = test_groups
    n_epochs = 200
    # ---- Initialise model and optimiser and scheduler
    model = neural_network_3(n_features, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimiser,
                              gamma=0.99)  # should be about 1/20 after 300 epochs
    model.train()
    train_stats_master_df = train_stats_master_df.sample(frac=1)  # shuffle
    x_tensor_train = torch.tensor(train_stats_master_df.iloc[:, :-1].values,
                                  dtype=torch.float).to(
        device).double()  # exclude last column
    y_tensor_train = torch.tensor(train_stats_master_df.iloc[:, -1].values,
                                  dtype=torch.float).to(
        device).long()
    x_tensor_eval = torch.tensor(eval_stats_master_df.iloc[:, :-1].values,
                                 dtype=torch.float).to(
        device).double()  # exclude last column
    y_tensor_eval = torch.tensor(eval_stats_master_df.iloc[:, -1].values,
                                 dtype=torch.float).to(
        device).long()

    train_losses = []
    eval_losses = []
    for epoch in range(n_epochs):
        print(f"Fold {i + 1} Epoch {epoch + 1}")
        # ---- training step
        correct = 0
        n_examples = 0
        all_preds = []
        all_labels = []

        # Forward pass
        model.train()
        logits = model(x_tensor_train)
        loss = criterion(logits, y_tensor_train)
        train_loss = loss.item()
        train_losses.append(train_loss)
        # Calculate metrics
        n_examples += y_tensor_train.size(0)  # size of the batch
        y_pred = logits.argmax(dim=1)
        all_preds.extend(y_pred.cpu().numpy())
        all_labels.extend(y_tensor_train.cpu().numpy())
        correct += (y_pred.round() == y_tensor_train).sum().item()  # number of correct items
        # Backward pass and optimization
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        labels_set = sorted(set(all_labels))
        cm = confusion_matrix(all_labels, all_preds, labels=labels_set)
        cm_df = pd.DataFrame(cm, index=labels_set, columns=labels_set)
        print(cm_df)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_set)
        # disp.plot()
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"TRAINING: accuracy={accuracy:.3f}, f1_score={f1:.3f}, "
              f"precision={precision:.3f}, recall={recall:.3f}")
        # print(classification_report(all_labels, all_preds, digits=3))
        # ---- validation step
        model.eval()  # put the model in evaluation mode
        correct = 0
        n_examples = 0
        all_preds = []
        all_labels = []

        # Forward pass
        logits = model(x_tensor_eval)
        loss = criterion(logits, y_tensor_eval)
        y_pred = logits.argmax(dim=1)
        eval_loss = loss.item()
        eval_losses.append(eval_loss)
        # Calculate training accuracy
        n_examples += y_tensor_eval.size(0)  # size of the batch
        correct += (y_pred.round() == y_tensor_eval).sum().item()  # number of correct items
        all_preds.extend(y_pred.cpu().numpy())
        all_labels.extend(y_tensor_eval.cpu().numpy())

        labels_set = sorted(set(all_labels))
        cm = confusion_matrix(all_labels, all_preds, labels=labels_set)
        cm_df = pd.DataFrame(cm, index=labels_set, columns=labels_set)
        print(cm_df)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_set)
        # disp.plot()
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)
        assert accuracy == correct / n_examples
        print(f"TEST: accuracy={accuracy:.3f}, f1_score={f1:.3f}, "
              f"precision={precision:.3f}, recall={recall:.3f}")
        # print(classification_report(all_labels, all_preds, digits=3))
        print(f"Fold {i + 1} Epoch {epoch + 1} Train Loss {train_loss} Eval Loss {eval_loss}")
    accuracy_list.append(accuracy)
    f1_score_list.append(f1_score)
    precision_list.append(precision)
    recall_list.append(recall)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.plot(eval_losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"train_loss_{i}.png")
    # plt.show()
    plt.close()
    torch.save(model.state_dict(),f"model_weights_{i}.pth")
mean_accuracy = np.mean(accuracy_list)
mean_f1 = np.mean(f1_score_list)
mean_precision = np.mean(precision_list)
mean_recall = np.mean(recall_list)
print(f"FINAL: accuracy={mean_accuracy:.2f}, f1_score={mean_f1:.2f}, "
      f"precision={mean_precision:.2f}, recall={mean_recall:.2f}")
