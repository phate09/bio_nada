import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, \
    accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging

from focal_loss import FocalLoss
from neural_network import neural_network_3


def generate_run_folder(prefix: str = "run"):
    # Get project root from Git
    git_root = Path(
        subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())

    # Timestamped folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = git_root / "runs" / f"{prefix}_{timestamp}"

    # Create it
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created: {output_dir}")
    return output_dir


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # File handler
    fh = logging.FileHandler('training.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def train_cell_model(config: dict, master_df: pd.DataFrame):
    logger = setup_logger()
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    logger.info(f"Start. Using {device}")
    lr = config["lr"]
    batch_size = config["batch_size"]
    n_epochs = config["epochs"]
    label_column = config["label_column"]
    output_dim = master_df[label_column].nunique()
    kfold_group_column = config["kfold_group_column"]
    colums_keep = config["feature_columns"] + [kfold_group_column] + [label_column]
    # Keep only the specified columns
    master_df = master_df[colums_keep]
    n_features = len(config["feature_columns"])
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    n_splits = config["n_splits"]
    lr_decay_factor = config["lr_decay_factor"]
    k_fold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
    if config["sampler"] == "RUS":
        rus = RandomUnderSampler(random_state=seed, replacement=False)
    elif config["sampler"] == "ROS":
        rus = RandomOverSampler(random_state=seed)
    else:
        rus = None
    if config["loss"] == "CE":
        criterion = nn.CrossEntropyLoss()
    elif config["loss"] == "Focal":
        criterion = FocalLoss(alpha=config["alpha"], gamma=config["gamma"])
    else:
        raise NotImplementedError()

    accuracy_list = []
    f1_score_list = []
    precision_list = []
    recall_list = []
    for i, (train_idx, test_idx) in enumerate(
        k_fold.split(master_df, master_df.loc[:, label_column], master_df.loc[:,
                                                                kfold_group_column])):  # k-fold
        train_groups = master_df.loc[train_idx]
        test_groups = master_df.loc[test_idx]
        train_groups.drop([kfold_group_column], axis=1, inplace=True)
        test_groups.drop([kfold_group_column], axis=1, inplace=True)
        if rus is not None:
            X_resampled, y_resampled = rus.fit_resample(train_groups,
                                                        train_groups.iloc[:, -1].values)
        else:
            X_resampled = train_groups
        train_stats_master_df = X_resampled
        eval_stats_master_df = test_groups

        # ---- Initialise model and optimiser and scheduler
        model = neural_network_3(n_features, output_dim).to(device)

        optimiser = optim.Adam(model.parameters(), lr=lr)

        scheduler = ExponentialLR(optimiser,
                                  gamma=lr_decay_factor)  # should be about 1/20 after 300 epochs

        train_stats_master_df = train_stats_master_df.sample(frac=1)  # shuffle
        x_tensor_train_total = torch.tensor(train_stats_master_df.iloc[:, :-1].values,
                                            dtype=torch.float)  # exclude last column
        y_tensor_train_total = torch.tensor(train_stats_master_df.iloc[:, -1].values,
                                            dtype=torch.float)
        x_tensor_eval_total = torch.tensor(eval_stats_master_df.iloc[:, :-1].values,
                                           dtype=torch.float)  # exclude last column
        y_tensor_eval_total = torch.tensor(eval_stats_master_df.iloc[:, -1].values,
                                           dtype=torch.float)
        # DataLoaders
        train_loader = DataLoader(TensorDataset(x_tensor_train_total, y_tensor_train_total),
                                  batch_size=batch_size,
                                  shuffle=True)
        eval_loader = DataLoader(TensorDataset(x_tensor_eval_total, y_tensor_eval_total),
                                 batch_size=batch_size, shuffle=False)

        train_losses = []
        eval_losses = []
        for epoch in range(n_epochs):
            logger.info(f"Fold {i + 1} Epoch {epoch + 1}")
            # ---- training step
            correct = 0
            n_examples = 0
            epoch_train_loss = 0.0
            num_train_batches = 0
            all_preds = []
            all_labels = []
            model.train()

            for X_batch, y_batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                # Load batch on GPU
                x_tensor_train = X_batch.to(device)
                y_tensor_train = y_batch.to(device).long()
                # Forward pass
                logits = model(x_tensor_train)
                loss = criterion(logits, y_tensor_train)
                train_loss = loss.item()
                epoch_train_loss += train_loss
                num_train_batches += 1
                # Calculate metrics
                n_examples += y_tensor_train.size(0)  # size of the batch
                y_pred = logits.argmax(dim=1)
                all_preds.extend(y_pred.cpu().numpy())
                all_labels.extend(y_tensor_train.cpu().numpy())
                correct += (
                    y_pred.round() == y_tensor_train).sum().item()  # number of correct items
                # Backward pass and optimization
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                scheduler.step()
            # After training, log metrics
            train_losses.append(epoch_train_loss / num_train_batches)
            labels_set = sorted(set(all_labels))
            cm = confusion_matrix(all_labels, all_preds, labels=labels_set)
            cm_df = pd.DataFrame(cm, index=labels_set, columns=labels_set)
            logger.info(cm_df)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')
            accuracy = accuracy_score(all_labels, all_preds)
            logger.info(f"TRAINING: accuracy={accuracy:.3f}, f1_score={f1:.3f}, "
                        f"precision={precision:.3f}, recall={recall:.3f}")
            logger.info(classification_report(all_labels, all_preds, digits=3))
            # ---- validation step
            with torch.no_grad():
                model.eval()  # put the model in evaluation mode
                correct = 0
                n_examples = 0
                epoch_eval_loss = 0.0
                num_eval_batches = 0
                all_preds = []
                all_labels = []
                for X_val, y_val in tqdm(eval_loader, desc=f"Evaluating Epoch {epoch + 1}"):
                    x_tensor_eval = X_val.to(device)
                    y_tensor_eval = y_val.to(device).long()
                    # Forward pass
                    logits = model(x_tensor_eval)
                    loss = criterion(logits, y_tensor_eval)
                    y_pred = logits.argmax(dim=1)
                    eval_loss = loss.item()
                    epoch_eval_loss += eval_loss
                    num_eval_batches += 1
                    # Calculate training accuracy
                    n_examples += y_tensor_eval.size(0)  # size of the batch
                    correct += (
                        y_pred.round() == y_tensor_eval).sum().item()  # number of correct items
                    all_preds.extend(y_pred.cpu().numpy())
                    all_labels.extend(y_tensor_eval.cpu().numpy())
                # After Validation, log metrics
                eval_losses.append(epoch_eval_loss / num_eval_batches)
                labels_set = sorted(set(all_labels))
                cm = confusion_matrix(all_labels, all_preds, labels=labels_set)
                cm_df = pd.DataFrame(cm, index=labels_set, columns=labels_set)
                logger.info(cm_df)
                precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_labels, all_preds, average='macro')
                f1 = f1_score(all_labels, all_preds, average='macro')
                accuracy = accuracy_score(all_labels, all_preds)
                assert accuracy == correct / n_examples
                logger.info(f"TEST: accuracy={accuracy:.3f}, f1_score={f1:.3f}, "
                            f"precision={precision:.3f}, recall={recall:.3f}")
                logger.info(classification_report(all_labels, all_preds, digits=3))
                logger.info(
                    f"Fold {i + 1} Epoch {epoch + 1} Train Loss {train_loss} Eval Loss {eval_loss}")
        accuracy_list.append(accuracy)
        f1_score_list.append(f1)
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
        torch.save(model.state_dict(), f"model_weights_{i}.pth")
    mean_accuracy = np.mean(accuracy_list)
    mean_f1 = np.mean(f1_score_list)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    logger.info(f"FINAL: accuracy={mean_accuracy:.2f}, f1_score={mean_f1:.2f}, "
                f"precision={mean_precision:.2f}, recall={mean_recall:.2f}")
