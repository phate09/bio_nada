import pickle
import random

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
    grouped_df_to_stats, get_dataframe_processed
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


full_df = get_dataframe()
with open("robustScaler.pkl",'rb') as f:
    scaler:RobustScaler=pickle.load(f)
with open("gaussianMixtureModel.pkl",'rb') as f:
    gaussian_model:GaussianMixture = pickle.load(f)
for group_name,df in full_df.groupby("id_random"):
    X_train = scaler.transform(df.iloc[:,0:6].values)
    cluster = gaussian_model.predict(X_train)
    print(group_name)
tsne_df = pd.DataFrame()
tsne_results = pca.fit_transform(X_train)
tsne_df['tsne-one'] = tsne_results[:,0]
tsne_df['tsne-two'] = tsne_results[:,1]
tsne_df['Label'] = gaussian_model.predict(X_train)
plt.figure(figsize=(16,10))
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
master_df[len(master_df.columns)]=full_df['Label']
