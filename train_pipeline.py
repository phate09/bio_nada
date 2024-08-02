from sklearn.model_selection import StratifiedKFold
from torch import nn, optim

from neural_network import get_simple_model
from pre_process import create_dataloaders, create_tensor_from_df, get_dataframe

model = get_simple_model()
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=1e-4)
k_fold = StratifiedKFold(n_splits=5, shuffle=True)
master_df = get_dataframe()
for train_idx, test_idx in k_fold.split(master_df, master_df.iloc[:, -1]):  # k-fold
    tensors = create_tensor_from_df(master_df.loc[train_idx], master_df.loc[test_idx])
    # use the * to reuse the individual variables in input
    train_loader, eval_loader = create_dataloaders(*tensors)
    n_epochs = 1
    train_accs = []
    train_losses = []
    for epoch in range(n_epochs):
        # ---- training step
        correct_train = 0
        total_train = 0
        train_loss = 0
        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            assert y_pred.shape == y_batch.shape
            train_loss += loss.item()
            # Calculate training accuracy
            total_train += y_batch.size(0)  # size of the batch
            correct_train += (y_pred.round() == y_batch).sum().item()  # number of correct items

            # Backward pass and optimization
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        print(f"Training accuracy = {correct_train / total_train:.2f}")
        # ---- validation step
        model.eval()
        correct_train = 0
        total_train = 0
        for X_batch, y_batch in eval_loader:
            # Forward pass
            y_pred = model(X_batch)
            # Calculate training accuracy
            total_train += y_batch.size(0)  # size of the batch
            correct_train += (y_pred.round() == y_batch).sum().item()  # number of correct items
        print(f"Validation accuracy = {correct_train / total_train:.2f}")
