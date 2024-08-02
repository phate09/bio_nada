import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataframe(data_folder: str = "data", label_file: str = "label.csv") -> pd.DataFrame:
    # ---- merge all data csv in dataframe
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

    # ----- append label column to dataframe
    label_df = pd.read_csv(label_file)

    master_df = master_df.merge(label_df, left_on="id_random", right_on="id_random", how="left")
    return master_df


def create_tensor_from_df(train_set_df: pd.DataFrame, test_set_df: pd.DataFrame) -> (pd.DataFrame,
                                                                                     pd.DataFrame,
                                                                                     pd.DataFrame,
                                                                                     pd.DataFrame):
    """Creates te tensors for train and test set starting from the dataframe
    :return x_tensor_train, y_tensor_train, x_tensor_test, y_tensor_test
    """
    x_tensor_train = torch.tensor(train_set_df.iloc[:, :-2].values,
                                  dtype=torch.float)  # exclude last two columns
    y_tensor_train = torch.tensor(train_set_df.iloc[:, -1].values, dtype=torch.float).reshape(-1, 1)
    assert x_tensor_train.shape[0] == y_tensor_train.shape[0]
    x_tensor_test = torch.tensor(test_set_df.iloc[:, :-2].values,
                                 dtype=torch.float)  # exclude last two columns
    y_tensor_test = torch.tensor(test_set_df.iloc[:, -1].values, dtype=torch.float).reshape(-1, 1)
    assert x_tensor_test.shape[0] == y_tensor_test.shape[0]
    return x_tensor_train, y_tensor_train, x_tensor_test, y_tensor_test


def create_dataloaders(x_tensor_train, y_tensor_train, x_tensor_test, y_tensor_test,
                       batch_size=1024) -> (DataLoader, DataLoader):
    train_loader = DataLoader(TensorDataset(x_tensor_train, y_tensor_train), batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(TensorDataset(x_tensor_test, y_tensor_test), batch_size=batch_size,
                             shuffle=False)
    return train_loader, test_loader