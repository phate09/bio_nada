import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


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
