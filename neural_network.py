import torch.nn as nn
import torch.optim as optim


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
