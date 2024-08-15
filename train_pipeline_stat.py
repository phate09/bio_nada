import random

import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import neural_network
from focal_loss import FocalLoss
from pre_process import create_dataloaders, create_tensor_from_df, get_dataframe, \
    grouped_df_to_stats, get_dataframe_processed
import progressbar
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device(
    'cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Start. Using {device}")

# criterion = nn.BCELoss()
criterion = FocalLoss()

k_fold = StratifiedKFold(n_splits=2, shuffle=True)
# rus = RandomUnderSampler(random_state=0, replacement=False)
rus = RandomOverSampler(random_state=0)

print("Preparing dataframe")
# Remember: last column is the label
master_df = get_dataframe_processed(label_file="some-lab.csv")

accuracy_list = []
f1_score_list_0 = []
precision_list_0 = []
recall_list_0 = []
f1_score_list_1 = []
precision_list_1 = []
recall_list_1 = []
for i, (train_idx, test_idx) in enumerate(k_fold.split(master_df, master_df.iloc[:, -1])):  # k-fold
    model = neural_network.neural_network_3(master_df.shape[1] - 1).to(device)  # reinitialise model
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    train_groups = master_df.loc[train_idx]
    test_groups = master_df.loc[test_idx]
    # X_resampled, y_resampled = rus.fit_resample(train_groups, train_groups.iloc[:, -1].values)
    X_resampled = train_groups
    n_epochs = 601
    train_accs = []
    train_losses = []
    train_stats_master_df = X_resampled
    eval_stats_master_df = test_groups
    for epoch in range(n_epochs):
        # ---- training step
        model.train()
        train_stats_master_df = train_stats_master_df.sample(frac=1)  # shuffle
        x_tensor_train = torch.tensor(train_stats_master_df.iloc[:, :-1].values,
                                      dtype=torch.float).to(
            device)  # exclude last two columns
        y_tensor_train = torch.tensor(train_stats_master_df.iloc[:, -1].values,
                                      dtype=torch.float).to(
            device)
        # train_loader = DataLoader(TensorDataset(x_tensor_train, y_tensor_train), batch_size=16,
        #                           shuffle=True)
        # train_loss = 0
        # for x_batch, y_batch in train_loader:
        y_pred = model(x_tensor_train).squeeze()
        loss = criterion(y_pred, y_tensor_train)
        # train_loss += loss.item()
        # Backward pass and optimization
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        y_true_array = y_tensor_train.cpu().detach().numpy()
        y_pred_array = y_pred.cpu().detach().numpy().round()
        # f1_score = metrics.f1_score(y_true_array, y_pred_array)
        # accuracy = metrics.accuracy_score(y_true_array, y_pred_array)
        # precision = metrics.precision_score(y_true_array, y_pred_array)
        # recall = metrics.recall_score(y_true_array, y_pred_array)
        if (epoch % 30) == 0:
            print(f"Fold {i + 1} Epoch {epoch}")  # +1 because i starts from 0
            # print(f"Train loss: {train_loss / len(train_loader)}")
            with torch.no_grad():
                print(classification_report(y_true_array, y_pred_array))
                # print(f"TRAINING: accuracy={accuracy:.2f}, f1_score={f1_score:.2f}, "
                #       f"precision={precision:.2f}, recall={recall:.2f}")
                # Precision: out of sick prediction, how many are truly sick
                # Recall: out of positives, how many are true positives (true positive rate)

                # ---- validation step
                model.eval()  # put the model in evaluation mode
                x_tensor_eval = torch.tensor(eval_stats_master_df.iloc[:, :-1].values,
                                             dtype=torch.float).to(
                    device)  # exclude last two columns
                y_tensor_eval = torch.tensor(eval_stats_master_df.iloc[:, -1].values,
                                             dtype=torch.float).to(
                    device)

                y_pred = model(x_tensor_eval).squeeze()
                y_true_array = y_tensor_eval.cpu().detach().numpy().astype(bool)
                y_pred_array = y_pred.cpu().detach().numpy().round().astype(bool)
                print(classification_report(y_true_array, y_pred_array))
                prec_0, rec_0, f1_score_0, _ = precision_recall_fscore_support(y_true_array,
                                                                               y_pred_array,
                                                                               pos_label=False,
                                                                               average='binary')
                prec_1, rec_1, f1_score_1, _ = precision_recall_fscore_support(y_true_array,
                                                                               y_pred_array,
                                                                               pos_label=True,
                                                                               average='binary')
                accuracy = metrics.accuracy_score(y_true_array, y_pred_array)
                # print(f"EVALUATION: accuracy={accuracy:.2f}, f1_score={f1_score:.2f}, "
                #       f"precision={precision:.2f}, recall={recall:.2f}")
    accuracy_list.append(accuracy)
    f1_score_list_0.append(f1_score_0)
    precision_list_0.append(prec_0)
    recall_list_0.append(rec_0)
    f1_score_list_1.append(f1_score_1)
    precision_list_1.append(prec_1)
    recall_list_1.append(rec_1)
print(
    f"MEAN EVALUATION accuracy={np.mean(accuracy_list):.2f}, f1_score_0={np.mean(f1_score_list_0):.2f}, "
    f"precision_0={np.mean(precision_list_0):.2f}, recall_0={np.mean(recall_list_0):.2f},"
    f" f1_score_1={np.mean(f1_score_list_1):.2f}, precision_1={np.mean(precision_list_1):.2f}, "
    f"recall_1={np.mean(recall_list_1):.2f}")
