from functools import partial
import ray
import torch
from ray import tune
from ray.air import RunConfig
from ray.tune import CLIReporter
from pprint import pprint
import pandas as pd

from pre_process import get_dataframe_processed

from train_methods import train

if __name__ == '__main__':
    ray.init(log_to_driver=False)
    device = torch.device('cpu')
    print(f"Start. Using {device}")
    pd.options.display.expand_frame_repr = False
    master_df = get_dataframe_processed(label_file="lab-21.csv")
    # creates a modified version of the function that preselect device and master_df
    train_partial = partial(train, device=device, master_df=master_df)
    metrics_columns = ["accuracy", "f1_score_0", "f1_score_1", "precision_0", "precision_1",
                       "recall_0", "recall_1"]
    run_config = RunConfig(verbose=0)

    search_space = {
        "loss": "Focal",
        "seeds": [0, 1, 2, 3, 4],
        "sampler": tune.grid_search(["None", "RUS", "ROS"]),
        "alpha": tune.grid_search([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "gamma": tune.grid_search([0.2, 0.5, 1, 2, 5]),
    }
    tuner = tune.Tuner(train_partial, param_space=search_space, run_config=run_config)
    results = tuner.fit()
    r1 = results.get_best_result(metric="f1_score_1", mode="max")
    print(f'OPTIMAL PARAMETERS for Focal Loss:{r1.config}')
    pprint(r1.metrics_dataframe[metrics_columns].iloc[0])
    filter_df_columns = ['config/' + x for x in list(search_space.keys())] + metrics_columns
    print(results.get_dataframe()[filter_df_columns])



    search_space = {
        "loss": "BCE",
        "seeds": [0],
        "sampler": tune.grid_search(["None", "RUS", "ROS"]),
        "alpha": 0,
        "gamma": 0,
    }
    tuner = tune.Tuner(train_partial, param_space=search_space, run_config=run_config)
    results = tuner.fit()
    r2 = results.get_best_result(metric="f1_score_1", mode="max")
    print(f'OPTIMAL PARAMETERS for BCE:')
    pprint(r2.config, width=1)
    pprint(r2.metrics_dataframe[metrics_columns].iloc[0])
    filter_df_columns = ['config/' + x for x in list(search_space.keys())] + metrics_columns
    print(results.get_dataframe()[filter_df_columns])
