import torch
from pprint import pprint
import train_methods
from pre_process import get_dataframe_processed, get_dataframe_processed_unsupervised
import pandas as pd

if __name__ == '__main__':
    config = {
        'loss': 'BCE',
        'seeds': [0],
        'sampler': 'RUS',
        'alpha': 0,
        'gamma': 0,
        'log_intermediate_results': False
    }
    device = torch.device('cpu')
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Start. Using {device}")
    # master_df = get_dataframe_processed(label_file="lab-21.csv")
    master_df = get_dataframe_processed_unsupervised(label_file="lab-21.csv")
    # master_df = get_dataframe_processed_unsupervised(label_file="label.csv")
    results = train_methods.train(config, master_df, device)
    print(pd.Series(results))
