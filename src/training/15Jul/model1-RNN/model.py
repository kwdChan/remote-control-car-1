from typing import Any, Callable, List, cast
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from IPython.display import display


class Model(nn.Module):
    def __init__(self):
        super().__init__()


        # from 64x114 -> 3520
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 16, (5, 5), stride=2), 
            nn.ReLU(), 
            nn.Conv2d(16, 32, (5, 5), stride=2), 
            nn.ReLU(), 
            nn.Conv2d(32, 64, (5, 5), stride=2), 
            nn.ReLU(), 
            nn.Flatten(),
            nn.Dropout(0.2), 
        )
        self.dense1 = nn.Linear(3520, 64)
        self.rnn1 = nn.RNN(64, 64)
        self.rnn2 = nn.RNN(64, 64)
        self.dense2 = nn.Linear(64, 4)


        self.relu = nn.LeakyReLU()

    

    def forward(self, x, h1=torch.zeros((1, 64)).to('cuda'), h2=torch.zeros((1, 64)).to('cuda')):
        x = self.conv_net(x)
        x = self.dense1(x)  # (seq_len, dense1_size)
        x = self.relu(x)

        # unbatched # (seq_len, dense1_size)
        x, h1 = self.rnn1(x, h1)
        x, h2 = self.rnn2(x, h2)
        x = self.dense2(x)


        return x, h1, h2