#!/usr/bin/env python
# coding: utf-8

# # Notebook Contents
# - Importing The Dependencies
# - Data Preparation
# - Bulding The NN
# - Evaluating The NN
# - Optimizing The Model
# - Saving The Best Model For Production

# ### `Importing The Dependencies`

# In[13]:


import pandas as pd
import numpy as np

import os

import matplotlib.pyplot as plt

import torch
from torch import nn
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import seaborn as sns

import torch

# Set the random seed for PyTorch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True  # If using CUDA for GPU acceleration
torch.backends.cudnn.benchmark = False

import numpy as np

# Set the random seed for NumPy
np.random.seed(0)

import random

random.seed(0)

# normalization
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# imbalance
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import classification_report, confusion_matrix

import csv

import warnings

warnings.filterwarnings("ignore")

# In[14]:


max_length = 399078 # the maximum number of rows


# In[15]:


# model
def model(max_lngth):
    '''
    This function for building the model architecture.
    '''
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(max_lngth * 6, 128), #todo maybe CNN
        nn.BatchNorm1d(128),  # BatchNorm layer
        nn.ReLU(),
        nn.Dropout(p=0.2),  # Dropout layer
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),  # BatchNorm layer
        nn.ReLU(),
        nn.Dropout(p=0.2),  # Dropout layer
        nn.Linear(64, 2),
        # nn.Softmax(dim=1)
    )

    return model


# In[16]:


criterion = nn.CrossEntropyLoss()


# In[17]:


# Let's start with training step

def train_step(train_loader, model, criterion, optimizer):
    correct_train = 0
    total_train = 0
    train_loss = 0
    for X_batch, y_batch in train_loader:
        # Convert the target labels to torch.LongTensor
        y_batch = y_batch.long()

        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        train_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(y_pred.data, 1)
        total_train += y_batch.size(0)
        correct_train += (predicted == y_batch).sum().item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = correct_train / total_train
    train_loss = train_loss / len(train_loader)

    return train_acc, train_loss


def val_step(model, criterion, val_loader):
    # Validation
    model.eval()
    with torch.no_grad():
        correct_val = 0
        total_val = 0
        val_loss = 0
        for X_val, y_val in val_loader:
            y_val = y_val.long()
            y_val_pred = model(X_val)
            val_loss += criterion(y_val_pred, y_val).item()

            _, predicted_val = torch.max(y_val_pred.data, 1)
            total_val += y_val.size(0)
            correct_val += (predicted_val == y_val).sum().item()

        val_acc = correct_val / total_val
        val_loss = val_loss / len(val_loader)

        return val_acc, val_loss


# In[18]:


import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import torch.optim as optim
import torch.nn as nn


def cross_validate_and_evaluate():
    n_epochs = 20
    batch_size = 8
    n_folds = 5
    # max_length=399078

    results = {'train_accs': [], 'val_accs': [], 'train_losses': [], 'val_losses': []}
    fold_metrics = {'precision_class_0': [], 'recall_class_0': [], 'f1-score_class_0': [],
                    'precision_class_1': [], 'recall_class_1': [], 'f1-score_class_1': []}

    for fold in range(n_folds):
        print("------------------------------")
        print(f"Fold {fold + 1}")
        print("------------------------------")

        #######################################################################
        # Concatenate all files in train set
        # Find lable
        labels_path = f'data-csv/fold_{fold + 1}/f1.csv'
        train_files_dir = f'data-csv/fold_{fold + 1}/train'

        # Read labels
        labels = pd.read_csv(labels_path)

        X_train = torch.empty((0, max_length, 6))
        y_train = torch.empty((0,), dtype=torch.int)

        for filee in os.listdir(
            train_files_dir):  # Loop through CSV files in the dynamically specified directory
            if filee.startswith("."):
                continue

            df = pd.read_csv(
                os.path.join(train_files_dir, filee))  # Read the CSV data into a DataFrame
            tensor = torch.tensor(df.values,
                                  dtype=torch.float)  # Ensure tensor is of type float for operations

            # Handle X
            if tensor.shape[0] < max_length:
                padding = torch.zeros((max_length - tensor.shape[0], 6))
                tensor = torch.cat((tensor, padding), dim=0)
            tensor = tensor.unsqueeze(0)  # Add an extra dimension at the beginning
            X_train = torch.cat((X_train, tensor), dim=0)

            # Handle y
            new_entry = torch.tensor(
                int(labels[labels['id_random'] == int(filee.split(".")[0])]['Label'])).unsqueeze(0)
            y_train = torch.cat((y_train, new_entry), dim=0)

        print(X_train.size(), y_train.size())

        ##########################################################################
        # Concatenate all files in test set and save the file name
        # Load the labels

        labels_test_path = f'data-csv/fold_{fold + 1}/f1-test.csv'
        test_files_dir = f'data-csv/fold_{fold + 1}/test'

        # Load the labels
        labelstest = pd.read_csv(labels_test_path)

        X_test = torch.empty((0, max_length, 6))
        y_test = torch.empty((0,), dtype=torch.int)  # Tensor for storing labels
        y_test_ids = torch.empty((0,), dtype=torch.int)  # Tensor for storing id_random values

        for filee in os.listdir(test_files_dir):
            if filee.startswith("."):
                continue

            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(test_files_dir, filee))

            # Convert the DataFrame to a PyTorch tensor
            tensor = torch.tensor(df.values,
                                  dtype=torch.float)  # Ensure tensor is of type float for operations

            # Pad the tensor if necessary
            if tensor.shape[0] < max_length:
                padding = torch.zeros((max_length - tensor.shape[0], 6))
                tensor = torch.cat((tensor, padding), dim=0)
            tensor = tensor.unsqueeze(0)  # Add an extra dimension at the beginning

            # Concatenate with the existing tensor
            X_test = torch.cat((X_test, tensor), dim=0)

            # Extract the id_random value and corresponding label
            id_random = int(filee.split(".")[0])
            label = int(labelstest[labelstest['id_random'] == id_random]['Label'].iloc[0])

            # Add the id_random and label to their respective tensors
            y_test_ids = torch.cat((y_test_ids, torch.tensor([id_random], dtype=torch.int)))
            y_test = torch.cat((y_test, torch.tensor([label], dtype=torch.int)))

        print(X_test.size(), y_test_ids.size(), y_test.size())
        #######################################################################
        # Concatenate all files in validation set
        # Find lable
        labels_val_path = f'data-csv/fold_{fold + 1}/f1-val.csv'
        val_files_dir = f'data-csv/fold_{fold + 1}/validation'

        labels = pd.read_csv(labels_val_path)

        X_val = torch.empty((0, max_length, 6))
        y_val = torch.empty((0,), dtype=torch.int)

        for filee in os.listdir(val_files_dir):  # Loop through CSV files
            if filee.startswith("."):
                continue
            # print(filee)a
            df = pd.read_csv(os.path.join(val_files_dir,
                                          filee))  # For each file, read the CSV data into a Pandas DataFrame (df).
            tensor = torch.tensor(
                df.values)  # Convert the DataFrame values to a PyTorch tensor (tensor).

            # Handle X
            if tensor.shape[
                0] < max_length:  # Check if the length of the tensor (number of rows) is less than max_length.
                padding = torch.zeros((max_length - tensor.shape[0],
                                       6))  # If so, pad the tensor with zeros along the first dimension to match max_length.
                tensor = torch.cat((tensor, padding), dim=0)
            tensor = tensor.unsqueeze(0)  # add an extra dimension at the beginning
            X_val = torch.cat((X_val, tensor), dim=0)

            # Handle y
            new_entry = torch.tensor(
                int(labels[labels['id_random'] == int(filee.split(".")[0])]['Label'])).unsqueeze(
                0)  # Extract the id_random value from the filename (filee) and use it to index the 'Label' from the labels DataFrame.
            y_val = torch.cat((y_val, new_entry),
                              dim=0)  # Convert the 'Label' value to a tensor and add an extra dimension at the beginning using .unsqueeze(0)

        print(X_val.size(), y_val.size())
        #######################################################################

        # print comment
        # Directly count occurrences if y_train_fold_res is already a numpy array
        class_counts = np.bincount(y_train)
        # class_counts will have the count of each class starting from 0 to n_classes-1
        num_class_0 = class_counts[0]
        num_class_1 = class_counts[1]
        # Calculate the total number of samples
        total_samples = num_class_0 + num_class_1
        print(f"Number of class 0 samples train: {num_class_0}")
        print(f"Number of class 1 samples train: {num_class_1}")
        print(f"Total number of samples train: {total_samples}")
        print("------------------------------")

        # Directly count occurrences if y_train_fold_res is already a numpy array
        class_counts = np.bincount(y_val)
        # class_counts will have the count of each class starting from 0 to n_classes-1
        num_class_0 = class_counts[0]
        num_class_1 = class_counts[1]
        # Calculate the total number of samples
        total_samples = num_class_0 + num_class_1
        print(f"Number of class 0 samples val: {num_class_0}")
        print(f"Number of class 1 samples val: {num_class_1}")
        print(f"Total number of samples val: {total_samples}")
        print("------------------------------")

        # print comment
        # Directly count occurrences if y_train_fold_res is already a numpy array
        class_counts = np.bincount(y_test)
        # class_counts will have the count of each class starting from 0 to n_classes-1
        num_class_0 = class_counts[0]
        num_class_1 = class_counts[1]
        # Calculate the total number of samples
        total_samples = num_class_0 + num_class_1
        print(f"Number of class 0 samples test: {num_class_0}")
        print(f"Number of class 1 samples test: {num_class_1}")
        print(f"Total number of samples test: {total_samples}")
        print("------------------------------")

        # Normalize data
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
        X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1))
        X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))
        print("normalization")
        # Oversampling
        ros = RandomOverSampler(random_state=0)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train.numpy())
        # X_train_res = X_train
        # y_train_res = y_train.numpy()
        print("balance")

        # print comment
        # Directly count occurrences if y_train_fold_res is already a numpy array
        class_counts = np.bincount(y_train_res)
        # class_counts will have the count of each class starting from 0 to n_classes-1
        num_class_0 = class_counts[0]
        num_class_1 = class_counts[1]
        # Calculate the total number of samples
        total_samples = num_class_0 + num_class_1
        print(f"Number of class 0 samples blance: {num_class_0}")
        print(f"Number of class 1 samples blance: {num_class_1}")
        print(f"Total number of samples blance: {total_samples}")
        print("------------------------------")

        # Convert back to tensors
        X_train_res = torch.tensor(X_train_res, dtype=torch.float32).reshape(-1, max_length,
                                                                             6)  # Adjust shape as necessary
        y_train_res = torch.tensor(y_train_res, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32).reshape(-1, max_length,
                                                                 6)  # Adjust shape as necessary
        y_val = torch.tensor(y_val, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, max_length,
                                                                   6)  # Adjust shape as necessary

        # DataLoaders
        train_loader = DataLoader(TensorDataset(X_train_res, y_train_res), batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size,
                                 shuffle=False)

        # Model and optimizer
        my_model = model(max_length)
        optimizer = optim.Adam(my_model.parameters(), lr=1e-4)

        # Training and validation for the current fold
        train_accs = []
        train_losses = []
        val_accs = []
        val_losses = []

        for epoch in range(n_epochs):
            train_acc, train_loss = train_step(train_loader, my_model, criterion, optimizer)
            val_acc, val_loss = val_step(my_model, criterion, val_loader)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch:{epoch + 1}/{n_epochs}')
            print(
                f'train_loss = {train_loss:.3f} | train_acc = {train_acc:.3f} | val_loss = {val_loss:.3f} | val_acc = {val_acc:.3f}')

        # Calculate the average metrics for the current fold
        avg_train_acc = sum(train_accs) / len(train_accs)
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_acc = sum(val_accs) / len(val_accs)
        avg_val_loss = sum(val_losses) / len(val_losses)

        # Append the aggregated metrics for the current fold to the results dictionary
        results['train_accs'].append(avg_train_acc)
        results['val_accs'].append(avg_val_acc)
        results['train_losses'].append(avg_train_loss)
        results['val_losses'].append(avg_val_loss)

        # Evaluate on validation set
        y_val_true, y_val_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = my_model(X_batch)
                _, predicted = torch.max(y_pred, 1)
                y_val_true.extend(y_batch.numpy())
                y_val_pred.extend(predicted.numpy())

        """"
        # Classification report and confusion matrix
        #print(classification_report(y_val_true, y_val_pred))
        plt.figure(figsize=(3, 3))
        sns.heatmap(confusion_matrix(y_val_true, y_val_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Fold {fold+1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        """

        # Evaluate the model on the test set

        print('---------Testing---------')
        y_true = []
        y_pred_list = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = my_model(X_batch)
                _, predicted = torch.max(y_pred.data, 1)
                y_true.extend(y_batch.numpy())
                y_pred_list.extend(predicted.numpy())

        # Calculate the classification report and confusion matrix
        print(classification_report(y_true, y_pred_list))

        # The confusion matrix
        plt.figure(figsize=(3, 3))
        sns.heatmap(data=confusion_matrix(y_true, y_pred_list),
                    annot=True,
                    cmap='viridis',
                    linecolor='k',
                    linewidths=1,
                    fmt='d')

        plt.title('Testing Confusion Matrix', c='r')
        plt.xlabel('Pred', color='r')
        plt.ylabel('Actual', color='r')
        plt.show()

        y_test_ids_numbers = [t.item() for t in y_test_ids]

        # Save the testing true and predicted labels, and test IDs to a CSV file
        with open(f'{fold + 1}-csv-under-15.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Name', 'TrueLabel', 'PredictedLabel'])  # Write header
            # Make sure all lists are of the same length
            data = zip(y_test_ids_numbers, y_true, y_pred_list)
            csv_writer.writerows(data)

        # Generate and store metrics
        report = classification_report(y_true=y_true, y_pred=y_pred_list, output_dict=True)

        # Store class-specific metrics
        fold_metrics['precision_class_0'].append(report['0']['precision'])
        fold_metrics['recall_class_0'].append(report['0']['recall'])
        fold_metrics['f1-score_class_0'].append(report['0']['f1-score'])

        fold_metrics['precision_class_1'].append(report['1']['precision'])
        fold_metrics['recall_class_1'].append(report['1']['recall'])
        fold_metrics['f1-score_class_1'].append(report['1']['f1-score'])

    # Calculate averages for each class
    avg_metrics = {metric: sum(values) / len(values) for metric, values in fold_metrics.items()}

    # Display the average metrics
    print('\nAverage Metrics Across All Folds:')
    print(f'Class 0 Precision: {avg_metrics["precision_class_0"] * 100:.3f}')
    print(f'Class 0 Recall: {avg_metrics["recall_class_0"] * 100:.3f}')
    print(f'Class 0 F1-Score: {avg_metrics["f1-score_class_0"] * 100:.3f}')
    print(f'Class 1 Precision: {avg_metrics["precision_class_1"] * 100:.3f}')
    print(f'Class 1 Recall: {avg_metrics["recall_class_1"] * 100:.3f}')
    print(f'Class 1 F1-Score: {avg_metrics["f1-score_class_1"] * 100:.3f}')

    # Convert fold_metrics to a DataFrame for easy saving
    metrics_df = pd.DataFrame(fold_metrics)

    # Optionally, you can add a row for averages
    metrics_df.loc['average'] = metrics_df.mean()

    # Save the DataFrame to a CSV file
    metrics_df.to_csv('fold_metrics_CSV-over-20.csv', index=True, index_label='Fold/Statistic')

    return results


# In[19]:


results = cross_validate_and_evaluate()

# In[20]:


import matplotlib.pyplot as plt
import numpy as np


def plot_fold_results(results):
    folds = np.arange(1, len(results['train_accs']) + 1)

    # Multiply accuracy and loss values by 100
    train_accs_percent = np.array(results['train_accs']) * 100
    val_accs_percent = np.array(results['val_accs']) * 100
    train_losses_percent = np.array(results['train_losses']) * 100
    val_losses_percent = np.array(results['val_losses']) * 100

    # Create a single plot with a single y-axis
    plt.figure(figsize=(10, 6))

    # Plot training and validation accuracy with different markers
    plt.plot(folds, train_accs_percent, 'o-', label='Training Accuracy', color='tab:blue')
    plt.plot(folds, val_accs_percent, 's-', label='Validation Accuracy', color='tab:orange')

    # Plot training and validation loss with different markers
    plt.plot(folds, train_losses_percent, 'o--', label='Training Loss', color='tab:green')
    plt.plot(folds, val_losses_percent, 's--', label='Validation Loss', color='tab:red')

    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy (%) / Loss (%)')
    plt.xticks(folds)
    plt.legend(loc='upper left')
    plt.title('Training and Validation Metrics per Fold')

    plt.show()


# Call the function with your results dictionary
plot_fold_results(results)

# In[ ]:


# In[ ]:


# In[ ]:



