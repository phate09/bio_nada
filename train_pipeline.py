from sklearn.model_selection import StratifiedKFold
from torch import nn, optim

from neural_network import get_simple_model
from pre_process import create_dataloaders, create_tensor_from_df, get_dataframe

model = get_simple_model()
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=1e-4)
k_fold = StratifiedKFold(n_splits=5, shuffle=True)
master_df = get_dataframe()
for i, (train_idx, test_idx) in enumerate(k_fold.split(master_df, master_df.iloc[:, -1])):  # k-fold
    tensors = create_tensor_from_df(master_df.loc[train_idx], master_df.loc[test_idx])
    # use the * to reuse the individual variables in input
    train_loader, eval_loader = create_dataloaders(*tensors)
    n_epochs = 1
    train_accs = []
    train_losses = []
    for epoch in range(n_epochs):
        # ---- training step
        correct = 0
        n_examples = 0
        train_loss = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            assert y_pred.shape == y_batch.shape
            train_loss += loss.item()
            # Calculate training accuracy
            n_examples += y_batch.size(0)  # size of the batch
            correct += (y_pred.round() == y_batch).sum().item()  # number of correct items
            true_positive += ((y_pred.round() == y_batch) & (y_batch == 1)).sum().item()
            true_negative += ((y_pred.round() == y_batch) & (y_batch == 0)).sum().item()
            false_positive += ((y_pred.round() != y_batch) & (y_batch == 1)).sum().item()
            false_negative += ((y_pred.round() != y_batch) & (y_batch == 0)).sum().item()
            # Backward pass and optimization
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        print(f"Fold {i + 1}")  # +1 because i starts from 0
        precision = true_positive / max((true_positive + false_positive), 1)  # prevents div by 0
        recall = true_positive / max((true_positive + false_negative), 1)  # prevents div by 0
        f1_score = 2 * (precision * recall) / max((precision + recall), 1)  # prevents div by 0
        accuracy = correct / n_examples
        print(f"TRAINING: accuracy={accuracy:.2f}, f1_score={f1_score:.2f}, "
              f"precision={precision:.2f}, recall={recall:.2f}")
        # ---- validation step
        model.eval()
        correct = 0
        n_examples = 0
        train_loss = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for X_batch, y_batch in eval_loader:
            # Forward pass
            y_pred = model(X_batch)
            # Calculate training accuracy
            n_examples += y_batch.size(0)  # size of the batch
            correct += (y_pred.round() == y_batch).sum().item()  # number of correct items
            true_positive += ((y_pred.round() == y_batch) & (y_batch == 1)).sum().item()
            true_negative += ((y_pred.round() == y_batch) & (y_batch == 0)).sum().item()
            false_positive += ((y_pred.round() != y_batch) & (y_batch == 1)).sum().item()
            false_negative += ((y_pred.round() != y_batch) & (y_batch == 0)).sum().item()
        precision = true_positive / max((true_positive + false_positive), 1)  # prevents div by 0
        recall = true_positive / max((true_positive + false_negative), 1)  # prevents div by 0
        f1_score = 2 * (precision * recall) / max((precision + recall), 1)  # prevents div by 0
        accuracy = correct / n_examples
        print(f"TEST: accuracy={accuracy:.2f}, f1_score={f1_score:.2f}, "
              f"precision={precision:.2f}, recall={recall:.2f}")
