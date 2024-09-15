import random

import numpy as np
import pandas as pd
import ray
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR

import neural_network
from focal_loss import FocalLoss


def train(config: dict, master_df: pd.DataFrame, device):
    intermediate_results_interval = 30
    log_intermediate_results = config["log_intermediate_results"] \
        if "log_intermediate_results" in config \
        else False
    accuracy_list = []
    f1_score_list_0 = []
    precision_list_0 = []
    recall_list_0 = []
    f1_score_list_1 = []
    precision_list_1 = []
    recall_list_1 = []
    assert config["seeds"]
    seeds = config["seeds"]
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # ray.train.torch.enable_reproducibility(seed=seed)
        if config["loss"] == "BCE":
            criterion = nn.BCELoss()
        elif config["loss"] == "Focal":
            criterion = FocalLoss(alpha=config["alpha"], gamma=config["gamma"])
        else:
            raise NotImplementedError()

        k_fold = StratifiedKFold(n_splits=2, shuffle=True,random_state=seed)
        if config["sampler"] == "RUS":
            rus = RandomUnderSampler(random_state=seed, replacement=False)
        elif config["sampler"] == "ROS":
            rus = RandomOverSampler(random_state=seed)
        else:
            rus = None

        for i, (train_idx, test_idx) in enumerate(
            k_fold.split(master_df, master_df.iloc[:, -1])):  # k-fold
            model = neural_network.soft_ordering_1dcnn_2(master_df.shape[1] - 1).to(
                device).double()  # reinitialise model
            optimiser = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = ExponentialLR(optimiser,
                                      gamma=0.995)  # should be about 1/20 after 600 epochs
            train_groups = master_df.loc[train_idx]
            test_groups = master_df.loc[test_idx]
            if rus is not None:
                X_resampled, y_resampled = rus.fit_resample(train_groups,
                                                            train_groups.iloc[:, -1].values)
            else:
                X_resampled = train_groups
            n_epochs = 600
            train_stats_master_df = X_resampled
            eval_stats_master_df = test_groups
            for epoch in range(n_epochs):
                # Training Step
                print_intermediate_results = (log_intermediate_results and
                                              (epoch % intermediate_results_interval) == 0)
                if print_intermediate_results:
                    print(f"Seed {seed} Fold {i + 1} Epoch {epoch}")  # +1 because i starts from 0
                    train_step(criterion, device, model, optimiser, scheduler,
                               train_stats_master_df,
                               print_report=True)
                    evaluation_step(device, eval_stats_master_df, model,
                                    print_report=True)
                else:
                    train_step(criterion, device, model, optimiser, scheduler,
                               train_stats_master_df,
                               print_report=False)

            # Final Epoch - Always Print report
            print(f"Seed {seed} Fold {i + 1} Epoch {n_epochs}")  # n_epochs because last
            train_step(criterion, device, model, optimiser, scheduler,
                       train_stats_master_df,
                       print_report=True)
            evaluation_results = evaluation_step(device, eval_stats_master_df, model,
                                                 print_report=True)
            accuracy, f1_score_0, f1_score_1, prec_0, prec_1, rec_0, rec_1 = evaluation_results

            # Log last evaluation
            accuracy_list.append(accuracy)
            f1_score_list_0.append(f1_score_0)
            precision_list_0.append(prec_0)
            recall_list_0.append(rec_0)
            f1_score_list_1.append(f1_score_1)
            precision_list_1.append(prec_1)
            recall_list_1.append(rec_1)
    results = {"accuracy": float(np.mean(accuracy_list)),
               "f1_score_0": float(np.mean(f1_score_list_0)),
               "f1_score_1": float(np.mean(f1_score_list_1)),
               "precision_0": float(np.mean(precision_list_0)),
               "precision_1": float(np.mean(precision_list_1)),
               "recall_0": float(np.mean(recall_list_0)),
               "recall_1": float(np.mean(recall_list_1))}
    return results


def evaluation_step(device, eval_stats_master_df, model, print_report=False):
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
        if print_report:
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
    return accuracy, f1_score_0, f1_score_1, prec_0, prec_1, rec_0, rec_1


def train_step(criterion, device, model, optimiser, scheduler, train_stats_master_df,
               print_report=False):
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
    if print_report:
        y_true_array = y_tensor_train.cpu().detach().numpy().astype(bool)
        y_pred_array = y_pred.cpu().detach().numpy().round().astype(bool)
        print(classification_report(y_true_array, y_pred_array))
