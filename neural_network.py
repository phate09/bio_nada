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
