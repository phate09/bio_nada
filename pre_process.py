import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import progressbar
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
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


def impute_nan(label_df: pd.DataFrame) -> pd.DataFrame:
    if 'age' in label_df.columns:
        return label_df.fillna({'age': label_df['age'].median(),
                                'trop': label_df['trop'].median(),
                                'ck': label_df['ck'].median(),
                                'egfr': label_df['egfr'].median(),
                                'chol': label_df['chol'].median(),
                                'htn': 0,
                                'dm': 0,
                                'mi': 0,
                                'pci': 0,
                                'cabg': 0,
                                'cva': 0,
                                'copd': 0,
                                'smoke': 0,
                                'ef_1': label_df['ef_1'].median()
                                })
    else:
        return label_df


def get_dataframe_processed(data_folder: str = "data",
                            label_file: str = "label.csv",
                            label_column: str = "Label") -> pd.DataFrame:
    preprocessed_file = Path(".preprocessed.csv")
    if not preprocessed_file.is_file():
        # ---- merge all data csv in dataframe
        pre_tensor = []
        pre_y_tensor = []
        label_df_original = pd.read_csv(label_file)
        # fill blanks in label_df
        label_df = impute_nan(label_df_original)
        for filee in progressbar.progressbar(os.listdir(
            data_folder),
            prefix="Preprocessing files"):  # Loop through CSV files in the dynamically specified directory
            if filee.startswith(".") or not filee.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(data_folder, filee))  # Read the CSV data into a DataFrame
            quantiles = 20
            quantile_list = []
            train_data_df = df

            for i in range(1, quantiles):
                quantile_list.append(train_data_df.quantile(i / quantiles))
            cv = train_data_df.std() / train_data_df.mean()
            range_value = train_data_df.max() - train_data_df.min()
            iqr = train_data_df.quantile(0.75) - train_data_df.quantile(0.25)
            statistics_df = pd.concat(
                [*quantile_list, train_data_df.mean(), train_data_df.std(),
                 train_data_df.kurtosis(),
                 train_data_df.skew(), cv, range_value, iqr, train_data_df.corr()], axis=1)
            id_random = int(filee.removesuffix(".csv"))
            y_row = label_df[label_df["id_random"] == id_random].iloc[0, :]
            y_label = y_row[label_column]
            y_data = y_row[(y_row.index != label_column) & (y_row.index != "id_random")]
            pre_tensor.append(np.concatenate([statistics_df.values.flatten(), y_data.values]))
            pre_y_tensor.append(y_label)
        master_train_data_df = pd.DataFrame(pre_tensor)
        master_y_data_df = pd.DataFrame(pre_y_tensor)
        master_df = pd.concat([master_train_data_df, master_y_data_df], axis=1)  # append lbl at end
        master_df.to_csv(preprocessed_file, header=False, index=False)
    else:
        master_df = pd.read_csv(preprocessed_file, index_col=False, header=None)
    return master_df


def get_dataframe_processed_unsupervised(data_folder: str = "data",
                                         label_file: str = "label.csv",
                                         label_column: str = "Label",
                                         scaler_file="robustScaler.pkl",
                                         clustering_ifle="gaussianMixtureModel.pkl") -> pd.DataFrame:
    preprocessed_file = Path(".preprocessed_unsupervised.csv")
    scaler_file = Path(scaler_file)
    unsupervised_model_file = Path(clustering_ifle)
    if not preprocessed_file.is_file():
        # ---- merge all data csv in dataframe
        pre_tensor = []
        pre_y_tensor = []
        label_df_original = pd.read_csv(label_file)
        # fill blanks in label_df
        label_df = impute_nan(label_df_original)
        with scaler_file.open(mode='rb') as f:
            scaler: RobustScaler = pickle.load(f)
        with unsupervised_model_file.open(mode='rb') as f:
            gaussian_model: GaussianMixture = pickle.load(f)
        for filee in progressbar.progressbar(os.listdir(
            data_folder),
            prefix="Preprocessing files"):  # Loop through CSV files in the dynamically specified directory
            if filee.startswith(".") or not filee.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(data_folder, filee))  # Read the CSV data into a DataFrame
            quantiles = 20
            quantile_list = []
            train_data_df = df  # pd.DataFrame(scaler.transform(df))
            for i in range(1, quantiles):
                quantile_list.append(train_data_df.quantile(i / quantiles))
            cv = train_data_df.std() / train_data_df.mean()
            range_value = train_data_df.max() - train_data_df.min()
            iqr = train_data_df.quantile(0.75) - train_data_df.quantile(0.25)
            train_data_rescaled = scaler.transform(train_data_df.values)
            train_data_cluster = pd.DataFrame(gaussian_model.predict(train_data_rescaled))
            statistics_df = pd.concat(
                [*quantile_list, train_data_df.mean(), train_data_df.std(),
                 train_data_df.kurtosis(),
                 train_data_df.skew(), cv, range_value, iqr, train_data_df.corr()], axis=1)
            value_counts = np.array(
                [train_data_cluster[train_data_cluster == x].count().item() for x in
                 range(gaussian_model.n_components)],
                dtype=np.int32)
            id_random = int(filee.removesuffix(".csv"))
            y_row = label_df[label_df["id_random"] == id_random].iloc[0, :]
            y_label = y_row[label_column]
            y_data = y_row[(y_row.index != label_column) & (y_row.index != "id_random")]
            pre_tensor.append(
                np.concatenate([statistics_df.values.flatten(), value_counts, y_data.values]))
            pre_y_tensor.append(y_label)
        master_train_data_df = pd.DataFrame(pre_tensor)
        master_y_data_df = pd.DataFrame(pre_y_tensor)
        master_df = pd.concat([master_train_data_df, master_y_data_df], axis=1)  # append lbl at end
        master_df.to_csv(preprocessed_file, header=False, index=False)
    else:
        master_df = pd.read_csv(preprocessed_file, index_col=False, header=None)
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


def grouped_df_to_stats(master_df, train_groups_names):
    pre_tensor = []
    pre_y_tensor = []
    for group in progressbar.progressbar(train_groups_names, prefix="Group"):
        df = master_df.get_group(group).sample(frac=1)
        quantiles = 20
        quantile_list = []
        train_data = df.iloc[:, :-2]
        y_data = df.iloc[0, -1]
        pre_y_tensor.append(y_data)
        for i in range(1, quantiles):
            quantile_list.append(train_data.quantile(i / quantiles))
        train_data.mean()
        train_data.std()

        cv = train_data.std() / train_data.mean()
        range_value = train_data.max() - train_data.min()
        iqr = train_data.quantile(0.75) - train_data.quantile(0.25)
        statistics_df = pd.concat(
            [*quantile_list, train_data.mean(), train_data.std(), train_data.kurtosis(),
             train_data.skew(), cv, range_value, iqr, train_data.corr()], axis=1)
        pre_tensor.append(statistics_df.values.flatten())
    stats_master_df = pd.DataFrame(pre_tensor)
    y_df = pd.DataFrame(pre_y_tensor)
    stats_master_df_result = pd.concat([stats_master_df, y_df], axis=1)
    return stats_master_df_result
