import pickle
import random
from pathlib import Path

import numpy as np
import sklearn.cluster
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from torch import nn, optim
import pickle

from neural_network import get_simple_model, neural_network_2, conv_neural_network, \
    neural_network_3, neural_network_4
from pre_process import create_dataloaders, create_tensor_from_df, get_dataframe, \
    grouped_df_to_stats, get_dataframe_processed, get_dataframe_processed_unsupervised
import progressbar
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from sklearn.manifold import TSNE
import seaborn as sns
import sklearn.cluster

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
pca = PCA(n_components=2)
grid_search = False  # it takes 1H to do a parameter scan, keep False
full_df = get_dataframe()
scaler_file = Path("robustScaler.pkl")
unsupervised_model_file = Path("gaussianMixtureModel.pkl")
X_train: pd.DataFrame  # dataframe holding the data
scaler: RobustScaler  # scaler
if not scaler_file.is_file() or not unsupervised_model_file.is_file():
    scaler = RobustScaler()
    gaussian_model: GaussianMixture
    X_train = scaler.fit_transform(full_df.iloc[:, 0:6].values)
    with scaler_file.open(mode='wb') as f:
        pickle.dump(scaler, f)
    if grid_search:
        param_grid = {
            "n_components": range(3, 10),
            "covariance_type": ["diag", "full"],  # "spherical", "tied",
        }


        def gmm_bic_score(estimator, X):
            """Callable to pass to GridSearchCV that will use the BIC score."""
            # Make it negative since GridSearchCV expects a score to maximize
            return -estimator.bic(X)


        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score, n_jobs=15
        )
        grid_search.fit(X_train)
        gaussian_model: GaussianMixture = grid_search.best_estimator_
        df = pd.DataFrame(grid_search.cv_results_)[
            ["param_n_components", "param_covariance_type", "mean_test_score"]
        ]
        df["mean_test_score"] = -df["mean_test_score"]
        df = df.rename(
            columns={
                "param_n_components": "Number of components",
                "param_covariance_type": "Type of covariance",
                "mean_test_score": "BIC score",
            }
        )
        df.sort_values(by="BIC score").head()
        import seaborn as sns

        sns.catplot(
            data=df,
            kind="bar",
            x="Number of components",
            y="BIC score",
            hue="Type of covariance",
        )
        plt.show()
    else:
        gaussian_model = GaussianMixture(n_components=8, covariance_type='diag')  # best combination
        gaussian_model.fit(X_train)
    with unsupervised_model_file.open(mode='wb') as f:
        pickle.dump(gaussian_model, f)
else:
    with scaler_file.open(mode='rb') as f:
        scaler: RobustScaler = pickle.load(f)
        X_train = scaler.transform(full_df.iloc[:, 0:6].values)
    with unsupervised_model_file.open(mode='rb') as f:
        gaussian_model: GaussianMixture = pickle.load(f)
visualise = False
if visualise:
    tsne_df = pd.DataFrame()
    tsne_results = pca.fit_transform(X_train)
    tsne_df['tsne-one'] = tsne_results[:, 0]
    tsne_df['tsne-two'] = tsne_results[:, 1]
    tsne_df['Label'] = gaussian_model.predict(X_train)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="Label",
        palette=sns.color_palette("hls", 6),
        data=tsne_df.sample(n=1000),
        legend="full",
        alpha=0.7
    )
    plt.show()
master_df = pd.DataFrame(X_train)
master_df[len(master_df.columns)] = full_df['Label']
get_dataframe_processed_unsupervised()