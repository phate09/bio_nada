import random
import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim

import neural_network
from focal_loss import FocalLoss
from pre_process import get_dataframe_processed
from sklearn import metrics

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device('cpu')
print(f"Start. Using {device}")

# Define alpha and gamma values for Focal Loss
alpha_values = [0.25,0.5,0.75, 0.8]
gamma_values = [1,2,3,4,5]

# Prepare dataframe
print("Preparing dataframe")
master_df = get_dataframe_processed(label_file="lab-21.csv")

# Placeholder lists to store results for different alpha and gamma combinations
results = []

k_fold = StratifiedKFold(n_splits=2, shuffle=True)

for alpha in alpha_values:
    for gamma in gamma_values:
        print(f"Running for alpha={alpha}, gamma={gamma}")
        
        # Reset seed before each combination of alpha and gamma
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Lists to store results for each fold
        accuracy_list = []
        f1_score_list_0 = []
        precision_list_0 = []
        recall_list_0 = []
        f1_score_list_1 = []
        precision_list_1 = []
        recall_list_1 = []

        # K-Fold cross-validation
        for i, (train_idx, test_idx) in enumerate(k_fold.split(master_df, master_df.iloc[:, -1])):
            # Initialize the model and optimizer for each fold and each combination of alpha and gamma
            #model = neural_network.neural_network_4(master_df.shape[1] - 1).to(device)
            model = neural_network.soft_ordering_1dcnn(input_dim=master_df.shape[1] - 1, output_dim=1).to(device)  # to use soft ordering
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = FocalLoss(alpha=alpha, gamma=gamma)

            train_groups = master_df.loc[train_idx]
            test_groups = master_df.loc[test_idx]
            X_resampled = train_groups

            n_epochs = 601
            train_stats_master_df = X_resampled
            eval_stats_master_df = test_groups

            # Train the model
            for epoch in range(n_epochs):
                model.train()
                train_stats_master_df = train_stats_master_df.sample(frac=1)  # shuffle
                x_tensor_train = torch.tensor(train_stats_master_df.iloc[:, :-1].values,
                                              dtype=torch.float).to(device)
                y_tensor_train = torch.tensor(train_stats_master_df.iloc[:, -1].values,
                                              dtype=torch.float).to(device)

                y_pred = model(x_tensor_train).squeeze()
                loss = criterion(y_pred, y_tensor_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch % 30) == 0:
                    print(f"Fold {i + 1}, Epoch {epoch}")

                    # Validation step
                    model.eval()
                    x_tensor_eval = torch.tensor(eval_stats_master_df.iloc[:, :-1].values,
                                                 dtype=torch.float).to(device)
                    y_tensor_eval = torch.tensor(eval_stats_master_df.iloc[:, -1].values,
                                                 dtype=torch.float).to(device)

                    y_pred = model(x_tensor_eval).squeeze()
                    y_true_array = y_tensor_eval.cpu().detach().numpy().astype(bool)
                    y_pred_array = y_pred.cpu().detach().numpy().round().astype(bool)
                    print(classification_report(y_true_array, y_pred_array))

                    # Calculate performance metrics
                    prec_0, rec_0, f1_score_0, _ = precision_recall_fscore_support(y_true_array, y_pred_array,
                                                                                   pos_label=False, average='binary')
                    prec_1, rec_1, f1_score_1, _ = precision_recall_fscore_support(y_true_array, y_pred_array,
                                                                                   pos_label=True, average='binary')
                    accuracy = metrics.accuracy_score(y_true_array, y_pred_array)

            # Append metrics for each fold
            accuracy_list.append(accuracy)
            f1_score_list_0.append(f1_score_0)
            precision_list_0.append(prec_0)
            recall_list_0.append(rec_0)
            f1_score_list_1.append(f1_score_1)
            precision_list_1.append(prec_1)
            recall_list_1.append(rec_1)

        # Store results for the current alpha and gamma
        results.append({
            'alpha': alpha,
            'gamma': gamma,
            'accuracy': np.mean(accuracy_list) * 100,
            'precision_0': np.mean(precision_list_0) * 100,
            'recall_0': np.mean(recall_list_0) * 100,
            'f1_score_0': np.mean(f1_score_list_0) * 100,
            'precision_1': np.mean(precision_list_1) * 100,
            'recall_1': np.mean(recall_list_1) * 100,
            'f1_score_1': np.mean(f1_score_list_1) * 100
        })

        # Print "TERMINATED" after each combination of alpha and gamma is tried
        print("TERMINATED")

# Print final results in a table format
#print("\nFinal Results (in %):")
#print(f"{'Alpha':<6}{'Gamma':<6}{'Accuracy':<10}{'Precision_0':<12}{'Recall_0':<10}{'F1_0':<8}{'Precision_1':<12}{'Recall_1':<10}{'F1_1':<8}")
#for res in results:
 #   print(f"{res['alpha']:<6}{res['gamma']:<6}{res['accuracy']:<10.0f}{res['precision_0']:<12.0f}{res['recall_0']:<10.0f}{res['f1_score_0']:<8.0f}"
  #        f"{res['precision_1']:<12.0f}{res['recall_1']:<10.0f}{res['f1_score_1']:<8.0f}")


# Print final results in a table format
print("\nFinal Results (in %):")
print(f"{'Alpha':<6}{'Gamma':<6}{'Accuracy':<10}{'Precision_0':<12}{'Recall_0':<10}{'F1_0':<8}{'Precision_1':<12}{'Recall_1':<10}{'F1_1':<8}")
for res in results:
    print(f"{res['alpha']:<6}{res['gamma']:<6}{res['accuracy']:<10.0f}{res['precision_0']:<12.0f}{res['recall_0']:<10.0f}{res['f1_score_0']:<8.0f}"
          f"{res['precision_1']:<12.0f}{res['recall_1']:<10.0f}{res['f1_score_1']:<8.0f}")