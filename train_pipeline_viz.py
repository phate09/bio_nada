import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim

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

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Start. Using {device}")

criterion = nn.BCELoss()

k_fold = StratifiedKFold(n_splits=20, shuffle=True)
# rus = RandomUnderSampler(random_state=0, replacement=False)
rus = RandomOverSampler(random_state=0)
print("Preparing dataframe")
master_df = get_dataframe_processed()
master_df.rename(columns={120:"y"},inplace=True)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(master_df.values)
master_df['pca-one'] = pca_result[:,0]
master_df['pca-two'] = pca_result[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=master_df,
    legend="full",
    alpha=0.3
)
plt.show()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(master_df.values)
master_df['tsne-one'] = tsne_results[:,0]
master_df['tsne-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=master_df,
    legend="full",
    alpha=0.3
)
plt.show()

full_df = get_dataframe()
tsne_results = tsne.fit_transform(full_df.iloc[:,0:6].values)
full_df['tsne-one'] = tsne_results[:,0]
full_df['tsne-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue="Label",
    palette=sns.color_palette("hls", 2),
    data=full_df,
    legend="full",
    alpha=0.3
)
plt.show()
