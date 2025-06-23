"""
Model for predicting whether the patient is sick or not.
Model1 + Model 2 -> Model3.
"""
import pandas as pd
import torch

import train_methods
from pre_process import get_dataframe_processed_with_fake_cell2

if __name__ == '__main__':
    pd.options.display.expand_frame_repr = False
    config = {
        'loss': 'BCE',
        'seeds': [0],
        'sampler': 'None', # RUS, ROS or None
        'alpha': 0,
        'gamma': 0,
        'log_intermediate_results': False
    }
    device = torch.device('cpu')
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Start. Using {device}")
    master_df = get_dataframe_processed_with_fake_cell2(label_file="lab-21.csv")
    results = train_methods.train(config, master_df, device)
    print(pd.Series(results))

