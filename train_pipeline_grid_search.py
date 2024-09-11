import random

import numpy as np
import ray
import torch
from ray import tune
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

import neural_network
from focal_loss import FocalLoss
from pre_process import create_dataloaders, create_tensor_from_df, get_dataframe, \
    grouped_df_to_stats, get_dataframe_processed, get_dataframe_processed_unsupervised
import progressbar
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device('cpu')#torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Start. Using {device}")

master_df = get_dataframe_processed(label_file="label.csv")
def train(config: dict):
    # if config["loss"] == "BCE":
    #     criterion = nn.BCELoss()
    # elif config["loss"] == "Focal":
    criterion = FocalLoss(alpha=config["alpha"], gamma=config["gamma"])
    # else:
    #     raise NotImplementedError()

    k_fold = StratifiedKFold(n_splits=2, shuffle=True)
    # rus = RandomUnderSampler(random_state=0, replacement=False)
    rus = RandomOverSampler(random_state=0)

    print("Preparing dataframe")
    # Remember: last column is the label
    # accuracy=0.72, f1_score_0=0.82, precision_0=0.82, recall_0=0.83, f1_score_1=0.31, precision_1=0.32, recall_1=0.30
    # MEAN EVALUATION accuracy=0.70, f1_score_0=0.80, precision_0=0.84, recall_0=0.76, f1_score_1=0.38, precision_1=0.33, recall_1=0.45

    # master_df = get_dataframe_processed_unsupervised(label_file="label.csv")

    accuracy_list = []
    f1_score_list_0 = []
    precision_list_0 = []
    recall_list_0 = []
    f1_score_list_1 = []
    precision_list_1 = []
    recall_list_1 = []
    for i, (train_idx, test_idx) in enumerate(
        k_fold.split(master_df, master_df.iloc[:, -1])):  # k-fold
        model = neural_network.soft_ordering_1dcnn(master_df.shape[1] - 1, output_dim=1).to(
            device).double()  # reinitialise model
        optimiser = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = ExponentialLR(optimiser, gamma=0.995)  # should be about 1/20 after 600 epochs
        train_groups = master_df.loc[train_idx]
        test_groups = master_df.loc[test_idx]
        if config["sampler"] == "RUS":
            X_resampled, y_resampled = rus.fit_resample(train_groups,
                                                        train_groups.iloc[:, -1].values)
        else:
            X_resampled = train_groups
        n_epochs = 601
        train_stats_master_df = X_resampled
        eval_stats_master_df = test_groups
        for epoch in range(n_epochs):
            # ---- training step
            model.train()
            train_stats_master_df = train_stats_master_df.sample(frac=1)  # shuffle
            x_tensor_train = torch.tensor(train_stats_master_df.iloc[:, :-1].values,
                                          dtype=torch.float).to(
                device).double()  # exclude last two columns
            y_tensor_train = torch.tensor(train_stats_master_df.iloc[:, -1].values,
                                          dtype=torch.float).to(
                device).double()
            y_pred = model(x_tensor_train).squeeze()
            loss = criterion(y_pred, y_tensor_train)
            # Backward pass and optimization
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
        with torch.no_grad():
            # ---- validation step
            model.eval()  # put the model in evaluation mode
            x_tensor_eval = torch.tensor(eval_stats_master_df.iloc[:, :-1].values,
                                         dtype=torch.float).to(
                device).double()  # exclude last two columns
            y_tensor_eval = torch.tensor(eval_stats_master_df.iloc[:, -1].values,
                                         dtype=torch.float).to(
                device).double()

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
        accuracy_list.append(accuracy)
        f1_score_list_0.append(f1_score_0)
        precision_list_0.append(prec_0)
        recall_list_0.append(rec_0)
        f1_score_list_1.append(f1_score_1)
        precision_list_1.append(prec_1)
        recall_list_1.append(rec_1)
        # print(
        #     f"MEAN EVALUATION accuracy={np.mean(accuracy_list):.2f}, "
        #     f"f1_score_0={np.mean(f1_score_list_0):.2f}, "
        #     f" f1_score_1={np.mean(f1_score_list_1):.2f},"
        #     f" precision_0={np.mean(precision_list_0):.2f},"
        #     f" precision_1={np.mean(precision_list_1):.2f},"
        #     f" recall_0={np.mean(recall_list_0):.2f},"
        #     f" recall_1={np.mean(recall_list_1):.2f}")
        results = {"accuracy": np.mean(accuracy_list),
                   "f1_score_0": np.mean(f1_score_list_0),
                   "f1_score_1": np.mean(f1_score_list_1),
                   "precision_0": np.mean(precision_list_0),
                   "precision_1": np.mean(precision_list_1),
                   "recall_0": np.mean(recall_list_0),
                   "recall_1": np.mean(recall_list_1)}
        return results


if __name__ == '__main__':
    # ray.init(local_mode=True)
    search_space = {
        "sampler": "None",
        "alpha": tune.grid_search([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
        "gamma": tune.grid_search([0,0.1,0.2,0.5,1,2,5]),
    }

    tuner = tune.Tuner(train, param_space=search_space)

    results = tuner.fit()
    r = results.get_best_result(metric="f1_score_1", mode="max")
    print(r.config)
