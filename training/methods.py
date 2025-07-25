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
from sklearn.model_selection import StratifiedGroupKFold, KFold, StratifiedKFold
from sklearn.utils import compute_class_weight
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import seaborn as sns
import torch
import io
from PIL import Image

from focal_loss import FocalLoss, FocalLossMulti
from neural_network import neural_network_3, neural_network_3_1


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


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Returns a matplotlib figure containing the plotted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    return fig


def log_confusion_matrix(writer, train_or_eval, y_true, y_pred, class_names, step):
    """Logs the confusion matrix as an image summary.
    train_or_eval should be either "train" or "eval".
    """
    fig = plot_confusion_matrix(y_true, y_pred, class_names)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0
    writer.add_image(f"ConfusionMatrix/{train_or_eval}", image[0], global_step=step)
    plt.close(fig)


def train_cell_model(config: dict, master_df: pd.DataFrame, run_folder: Path):
    logger = setup_logger()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    logger.info(f"Start. Using {device}")
    lr = config["lr"]
    batch_size = config["batch_size"]
    n_epochs = config["epochs"]
    label_column = config["label_column"]
    output_dim = master_df[label_column].nunique()
    kfold_group_column = config["kfold_group_column"]
    grouped_kfold = kfold_group_column is not None
    feature_columns = config["feature_columns"]
    if grouped_kfold:
        colums_keep = feature_columns + [kfold_group_column] + [label_column]
    else:
        colums_keep = feature_columns + [label_column]
    # Keep only the specified columns
    master_df = master_df[colums_keep]
    n_features = len(feature_columns)
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    n_splits = config["n_splits"]
    lr_decay_factor = config["lr_decay_factor"]
    if grouped_kfold:
        k_fold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
    else:
        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    if config["sampler"] == "RUS":
        rus = RandomUnderSampler(random_state=seed, replacement=False)
    elif config["sampler"] == "ROS":
        rus = RandomOverSampler(random_state=seed)
    else:
        rus = None
    if config["loss"] == "CE":
        y = master_df.loc[:, label_column]
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float, device=device))
    elif config["loss"] == "Focal":
        criterion = FocalLossMulti(alpha=torch.tensor(config["alpha"], dtype=torch.float, device=device).detach(), gamma=config["gamma"])
    else:
        raise NotImplementedError()

    accuracy_list = []
    f1_score_list = []
    precision_list = []
    recall_list = []
    if grouped_kfold:
        fold_iterator = k_fold.split(master_df, master_df.loc[:, label_column],
                                     master_df.loc[:, kfold_group_column])
    else:
        fold_iterator = k_fold.split(master_df, master_df.loc[:, label_column])
    for fold_idx, (train_idx, test_idx) in enumerate(fold_iterator):  # k-fold
        train_groups = master_df.loc[train_idx]
        test_groups = master_df.loc[test_idx]
        if grouped_kfold:
            # remove the columns used to perform stratified k-fold
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

        scheduler = ExponentialLR(optimiser, gamma=lr_decay_factor)

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
        # Create a writer for each fold
        writer = SummaryWriter(log_dir=str(run_folder / f"fold_{fold_idx + 1}"))
        writer.add_text("config", str(config))
        train_losses = []
        eval_losses = []
        for epoch in range(n_epochs):
            logger.info(f"Fold {fold_idx + 1} Epoch {epoch + 1}")
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
            train_loss_epoch_avg = epoch_train_loss / num_train_batches
            train_losses.append(train_loss_epoch_avg)
            labels_set = sorted(set(all_labels))
            cm = confusion_matrix(all_labels, all_preds, labels=labels_set)
            cm_df = pd.DataFrame(cm, index=labels_set, columns=labels_set)
            logger.info(cm_df)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            precision_per_class = precision_score(all_labels, all_preds, average=None,
                                                  zero_division=0, labels=labels_set)
            recall = recall_score(all_labels, all_preds, average='macro')
            recall_per_class = recall_score(all_labels, all_preds, average=None, labels=labels_set)
            f1 = f1_score(all_labels, all_preds, average='macro')
            f1_per_class = f1_score(all_labels, all_preds, average=None, labels=labels_set)
            accuracy = accuracy_score(all_labels, all_preds)
            logger.info(f"TRAINING: accuracy={accuracy:.3f}, f1_score={f1:.3f}, "
                        f"precision={precision:.3f}, recall={recall:.3f}")
            logger.info(classification_report(all_labels, all_preds, digits=3))
            writer.add_scalar("Loss/train", train_loss_epoch_avg, epoch)
            writer.add_scalar("Accuracy/train", accuracy, epoch)
            writer.add_scalar("Precision/train", precision, epoch)
            writer.add_scalar("Recall/train", recall, epoch)
            writer.add_scalar("F1/train", f1, epoch)

            for i, p in enumerate(precision_per_class):
                writer.add_scalar(f"Precision_class_{i}/train", p, epoch)
            for i, r in enumerate(recall_per_class):
                writer.add_scalar(f"Recall_class_{i}/train", r, epoch)
            for i, f in enumerate(f1_per_class):
                writer.add_scalar(f"F1_class_{i}/train", f, epoch)

            log_confusion_matrix(writer, "train", all_labels, all_preds, labels_set, epoch)
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
                eval_loss_epoch_avg = epoch_eval_loss / num_eval_batches
                eval_losses.append(eval_loss_epoch_avg)
                labels_set = sorted(set(all_labels))
                cm = confusion_matrix(all_labels, all_preds, labels=labels_set)
                cm_df = pd.DataFrame(cm, index=labels_set, columns=labels_set)
                logger.info(cm_df)
                precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                precision_per_class = precision_score(all_labels, all_preds, average=None,
                                                      zero_division=0, labels=labels_set)
                recall = recall_score(all_labels, all_preds, average='macro')
                recall_per_class = recall_score(all_labels, all_preds, average=None,
                                                labels=labels_set)
                f1 = f1_score(all_labels, all_preds, average='macro')
                f1_per_class = f1_score(all_labels, all_preds, average=None, labels=labels_set)
                accuracy = accuracy_score(all_labels, all_preds)
                assert accuracy == correct / n_examples
                logger.info(f"TEST: accuracy={accuracy:.3f}, f1_score={f1:.3f}, "
                            f"precision={precision:.3f}, recall={recall:.3f}")
                logger.info(classification_report(all_labels, all_preds, digits=3))
                logger.info(
                    f"Fold {fold_idx + 1} Epoch {epoch + 1} Train Loss {train_loss_epoch_avg} Eval Loss {eval_loss_epoch_avg}")
                
                writer.add_scalar("Loss/eval", eval_loss_epoch_avg, epoch)
                writer.add_scalar("Accuracy/eval", accuracy, epoch)
                writer.add_scalar("Precision/eval", precision, epoch)
                writer.add_scalar("Recall/eval", recall, epoch)
                writer.add_scalar("F1/eval", f1, epoch)

                for i, p in enumerate(precision_per_class):
                    writer.add_scalar(f"Precision_class_{i}/eval", p, epoch)
                for i, r in enumerate(recall_per_class):
                    writer.add_scalar(f"Recall_class_{i}/eval", r, epoch)
                for i, f in enumerate(f1_per_class):
                    writer.add_scalar(f"F1_class_{i}/eval", f, epoch)
                log_confusion_matrix(writer, "eval", all_labels, all_preds, labels_set, epoch)
        writer.close()  # close the writer at the end of the epoch
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
        plt.savefig(run_folder / f"train_loss_{fold_idx}.png")
        # plt.show()
        plt.close()
        torch.save(model.state_dict(), run_folder / f"model_weights_{fold_idx}.pth")
    mean_accuracy = np.mean(accuracy_list)
    mean_f1 = np.mean(f1_score_list)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    logger.info(f"FINAL: accuracy={mean_accuracy:.2f}, f1_score={mean_f1:.2f}, "
                f"precision={mean_precision:.2f}, recall={mean_recall:.2f}")
    writer = SummaryWriter(log_dir=str(run_folder / "fold_all"))
    writer.add_scalar("Accuracy/eval", mean_accuracy, 0)
    writer.add_scalar("Precision/eval", mean_precision, 0)
    writer.add_scalar("Recall/eval", mean_recall, 0)
    writer.add_scalar("F1/eval", mean_f1, 0)
    writer.close()

    # Save final model weights
    torch.save({
        'fold_models': [torch.load(run_folder / f"model_weights_{i}.pth") for i in range(n_splits)],
        'model_config': {
            'n_features': n_features,
            'output_dim': output_dim
        }
    }, run_folder / "model_weights_all.pth")

    return mean_accuracy, mean_f1, mean_precision, mean_recall
