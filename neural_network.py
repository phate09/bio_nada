import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

from soft_ordering_1dcnn import SoftOrdering1DCNN


def get_simple_model():
    model = nn.Sequential(nn.Linear(6, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 1),  # just 1 output because of 2 classes
                          nn.Sigmoid()  # just sigmoid instead of softmax
                          )
    return model


def neural_network_2():
    model = nn.Sequential(nn.Linear(6, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 64),
                          nn.ReLU(),
                          nn.Linear(64, 1),  # just 1 output because of 2 classes
                          nn.Sigmoid()  # just sigmoid instead of softmax
                          )
    return model


def neural_network_3(input_size):
    model = nn.Sequential(nn.Linear(input_size, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 64),
                          nn.ReLU(),
                          nn.Linear(64, 1),  # just 1 output because of 2 classes
                          nn.Sigmoid()  # just sigmoid instead of softmax
                          )
    return model


def neural_network_4(input_size):
    model = nn.Sequential(nn.Linear(input_size, 128),
                          nn.ReLU(),
                          nn.Dropout(p=0.2),
                          nn.BatchNorm1d(128),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.BatchNorm1d(64),
                          nn.Linear(64, 64),
                          nn.ReLU(),
                          nn.Linear(64, 1),  # just 1 output because of 2 classes
                          nn.Sigmoid()  # just sigmoid instead of softmax
                          )
    return model


def neural_network_5(input_size):
    model = nn.Sequential(nn.Linear(input_size, 64),
                          nn.Tanh(),
                          nn.Linear(64, 64),
                          nn.Tanh(),
                          nn.Linear(64, 1),  # just 1 output because of 2 classes
                          nn.Sigmoid()  # just sigmoid instead of softmax
                          )
    return model


def neural_network_6(input_size):
    model = nn.Sequential(nn.Linear(input_size, 128),
                          nn.LeakyReLU(),
                          nn.Linear(128, 64),
                          nn.LeakyReLU(),
                          nn.Linear(64, 64),
                          nn.LeakyReLU(),
                          nn.Linear(64, 1),  # just 1 output because of 2 classes
                          nn.Sigmoid()  # just sigmoid instead of softmax
                          )
    return model


def neural_network_tabular(input_size):
    model = nn.Sequential(nn.Linear(input_size, 128),
                          nn.BatchNorm1d(128),
                          nn.ReLU(),
                          nn.Dropout(0.3),

                          nn.Linear(128, 64),
                          nn.BatchNorm1d(64),
                          nn.ReLU(),
                          nn.Dropout(0.3),

                          nn.Linear(64, 32),
                          nn.BatchNorm1d(32),
                          nn.ReLU(),
                          nn.Dropout(0.3),

                          nn.Linear(32, 1),  # Single output for binary classification
                          nn.Sigmoid()  # Sigmoid for binary classification
                          )
    return model


def neural_network_7(input_size):
    model = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.BatchNorm1d(128),  # Batch Normalization
        nn.LeakyReLU(negative_slope=0.01),  # Leaky ReLU for better gradient flow
        nn.Dropout(p=0.3),  # Dropout with probability 0.3 to prevent overfitting

        nn.Linear(128, 64),
        nn.BatchNorm1d(64),  # Batch Normalization
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(p=0.3),  # Dropout

        nn.Linear(64, 32),
        nn.BatchNorm1d(32),  # Batch Normalization
        nn.LeakyReLU(negative_slope=0.01),
        nn.Dropout(p=0.3),  # Dropout

        nn.Linear(32, 1),  # Output layer
        nn.Sigmoid()  # Sigmoid for binary classification
    )
    return model


def conv_neural_network():
    model = nn.Sequential(nn.Conv1d(6, 128, kernel_size=5),
                          nn.ReLU(),
                          nn.Conv1d(128, 32, kernel_size=3),
                          nn.ReLU(),
                          nn.Conv1d(32, 1, kernel_size=1),
                          nn.ReLU(),
                          nn.Flatten(),
                          nn.AdaptiveMaxPool1d(1),  # just 1 output because of 2 classes
                          nn.Sigmoid()  # just sigmoid instead of softmax
                          )
    return model


import torch.nn as nn
import torch


def soft_ordering_1dcnn_2(input_dim, output_dim=1, sign_size=32, cha_input=16, cha_hidden=32, K=2,
                          dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
    model = SoftOrdering1DCNN(input_dim=input_dim, output_dim=output_dim, sign_size=sign_size,
                              cha_input=cha_input, cha_hidden=cha_hidden, K=K,
                              dropout_input=dropout_input, dropout_hidden=dropout_hidden,
                              dropout_output=dropout_output)
    return model


def soft_ordering_1dcnn(input_dim, output_dim=1, sign_size=32, cha_input=16, cha_hidden=32, K=2,
                        dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
    hidden_size = sign_size * cha_input
    sign_size1 = sign_size
    sign_size2 = sign_size // 2
    output_size = (sign_size // 4) * cha_hidden

    model = nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Dropout(dropout_input),
        torch.nn.utils.parametrizations.weight_norm(nn.Linear(input_dim, hidden_size, bias=False)),
        nn.CELU(),

        ReshapeLayer((-1, cha_input, sign_size1)),  # Custom reshape layer for input to conv layers

        # 1st conv layer
        nn.BatchNorm1d(cha_input),
        torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(cha_input, cha_input * K, kernel_size=5, stride=1, padding=2,
                                       groups=cha_input, bias=False)),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(output_size=sign_size2),

        # 2nd conv layer
        nn.BatchNorm1d(cha_input * K),
        nn.Dropout(dropout_hidden),
        torch.nn.utils.parametrizations.weight_norm(
            nn.Conv1d(cha_input * K, cha_hidden, kernel_size=3, stride=1, padding=1, bias=False)),
        nn.ReLU(),

        # 3rd conv layer
        nn.BatchNorm1d(cha_hidden),
        nn.Dropout(dropout_hidden),
        torch.nn.utils.parametrizations.weight_norm(
            nn.Conv1d(cha_hidden, cha_hidden, kernel_size=3, stride=1, padding=1, bias=False)),
        nn.ReLU(),

        # 4th conv layer
        nn.BatchNorm1d(cha_hidden),
        torch.nn.utils.parametrizations.weight_norm(
            nn.Conv1d(cha_hidden, cha_hidden, kernel_size=5, stride=1, padding=2, groups=cha_hidden,
                      bias=False)),
        nn.ReLU(),
        nn.AvgPool1d(kernel_size=4, stride=2, padding=1),

        nn.Flatten(),  # Flatten for fully connected layers

        nn.BatchNorm1d(output_size),
        nn.Dropout(dropout_output),
        torch.nn.utils.parametrizations.weight_norm(nn.Linear(output_size, output_dim, bias=False)),
        nn.Sigmoid()  # Sigmoid for binary classification
    )

    return model


def tabnet_sequential(input_dim, output_dim, sign_size=32, cha_input=16, cha_hidden=32, K=2,
                      dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
    hidden_size = sign_size * cha_input
    sign_size1 = sign_size
    sign_size2 = sign_size // 2
    output_size = (sign_size // 4) * cha_hidden

    model = nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Dropout(dropout_input),
        torch.nn.utils.parametrizations.weight_norm(nn.Linear(input_dim, hidden_size, bias=False)),
        nn.CELU(),

        ReshapeLayer((-1, cha_input, sign_size1)),  # Custom reshape layer for input to conv layers

        # 1st conv layer
        nn.BatchNorm1d(cha_input),
        torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(cha_input, cha_input * K, kernel_size=5, stride=1, padding=2,
                                       groups=cha_input, bias=False)),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(output_size=sign_size2),

        # 2nd conv layer
        nn.BatchNorm1d(cha_input * K),
        nn.Dropout(dropout_hidden),
        torch.nn.utils.parametrizations.weight_norm(
            nn.Conv1d(cha_input * K, cha_hidden, kernel_size=3, stride=1, padding=1, bias=False)),
        nn.ReLU(),

        # 3rd conv layer
        nn.BatchNorm1d(cha_hidden),
        nn.Dropout(dropout_hidden),
        torch.nn.utils.parametrizations.weight_norm(
            nn.Conv1d(cha_hidden, cha_hidden, kernel_size=3, stride=1, padding=1, bias=False)),
        nn.ReLU(),

        # 4th conv layer
        nn.BatchNorm1d(cha_hidden),
        torch.nn.utils.parametrizations.weight_norm(
            nn.Conv1d(cha_hidden, cha_hidden, kernel_size=5, stride=1, padding=2, groups=cha_hidden,
                      bias=False)),
        nn.ReLU(),
        nn.AvgPool1d(kernel_size=4, stride=2, padding=1),

        nn.Flatten(),  # Flatten for fully connected layers

        nn.BatchNorm1d(output_size),
        nn.Dropout(dropout_output),
        torch.nn.utils.parametrizations.weight_norm(nn.Linear(output_size, output_dim, bias=False)),
        nn.Sigmoid()  # Sigmoid for binary classification
    )

    return model


# Utility layer to reshape the tensor inside nn.Sequential
class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)
