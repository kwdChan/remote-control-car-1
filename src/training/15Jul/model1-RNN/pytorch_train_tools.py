from typing import Any, Callable, List, cast
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, BatchSampler
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from IPython.display import display

def temp_rnn_fit_step(model: nn.Module, dataloader: DataLoader, optimiser: optim.Optimizer, loss_fn, silent=True):
    size = len(dataloader.dataset) # type: ignore
    num_batches = len(dataloader)
    device = next(model.parameters()).device
    model.train()

    train_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        y_pred, _, _ = model(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        loss = loss.item()
        train_loss += loss

        if (not silent) and (not (batch % 100)):
            current = (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches  # TODO: this is not right
    return cast(float, train_loss)




def fit_step(model: nn.Module, dataloader: DataLoader, optimiser: optim.Optimizer, loss_fn, silent=True):
    size = len(dataloader.dataset) # type: ignore
    num_batches = len(dataloader)
    device = next(model.parameters()).device
    model.train()

    train_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        loss = loss.item()
        train_loss += loss

        if (not silent) and (not (batch % 100)):
            current = (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches  # TODO: this is not right
    return cast(float, train_loss)

def batch_predict(model, dataloader, progress_bar=True):
    device = next(model.parameters()).device
    model.train()

    result = []
    to_iter = enumerate(dataloader)
    if progress_bar: to_iter = tqdm(to_iter)

    for batch, (x, y) in to_iter:
        x, y = x.to(device), y.to(device)
        
        y_pred = model(x)
        result.append(y_pred.cpu())
    
    return torch.concat(result, axis=0)  # type: ignore
        
def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn, metrics: List=[], silent=True):
    device = next(model.parameters()).device
    num_batches = len(dataloader)
    model.eval()

    eval_loss = 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            print(batch, end='                 \r')
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            eval_loss += loss_fn(y_pred, y).item()

        # TODO: metrics
        
    eval_loss /= num_batches # TODO: this is not right

    if not silent: 
        print(f'eval_loss: {eval_loss}')
    return cast(float, eval_loss)

EpochCallback = Callable[[int, float, float], Any]



def _fit(
        model: nn.Module, 
        train_dataloader: DataLoader, 
        val_dataloader:DataLoader, 
        optimiser: optim.Optimizer, 
        loss_fn, epochs: int, 
        epoch_callbacks:List[EpochCallback], 
        finish_callback: List, 
        silent=True
    ):

    assert epochs
    for _epoch in range(epochs):
        train_loss = fit_step(model, train_dataloader, optimiser, loss_fn, silent=silent)
        val_loss = evaluate(model, val_dataloader, loss_fn, silent=silent)
        
        for cb in epoch_callbacks: 
            cb(_epoch, train_loss, val_loss)

    for cb in finish_callback:
        cb(train_loss, val_loss)  # type: ignore 



def fit(        
        model: nn.Module, 
        train_dataloader: DataLoader, 
        val_dataloader:DataLoader, 
        optimiser: optim.Optimizer, 
        loss_fn, 
        epochs: int, 
        epoch_callbacks:List[EpochCallback]=[], 
        finish_callback: List=[], 
    ):
    fig = UpdatingPlotlyLines('epoch', ['train_loss', 'val_loss'])
    fig.display()
    _fit(        
        model, 
        train_dataloader, 
        val_dataloader, 
        optimiser, 
        loss_fn, epochs, 
        epoch_callbacks+[fig.append], 
        finish_callback, 
        silent=True
    )
    return fig
    


class UpdatingPlotlyLines:
    def __init__(self, xname:str, keys: List[str]):
        self.xname = xname
        
        self.keys = keys

        self.fig = go.FigureWidget([go.Scatter({'x':[], 'y':[]}, name=k) for k in keys])

        self.data_x = {k:[] for k in keys}
        self.data_y = {k:[] for k in keys}

    def append(self, *args, **kwargs):
        kwargs = dict(kwargs, **dict(zip([self.xname]+self.keys, args)))
        for k in self.keys:
            self.data_x[k].append(kwargs[self.xname])
            self.data_y[k].append(kwargs[k])

        for idx, k in enumerate(self.keys):
            self.fig.data[idx].x = self.data_x[k] # type: ignore
            self.fig.data[idx].y = self.data_y[k] # type: ignore


    def display(self):
        display(self.fig)

def to_windowed_dataset(x, y, win_size, drop_last=True):
    """
    
    x and y are np.ndarray with the first dimension being the sample/sequence
    (100, ...) with win_size=5 -> 20 samples of (5, ...)

    """
    xform = lambda x: to_float32(np.stack(x, axis=0))
    x = list(BatchSampler(x, win_size, drop_last))
    y = list(BatchSampler(y, win_size, drop_last))
    return to_dataset(x, y, xform, xform)



to_float32 = lambda x: torch.from_numpy(x.astype(np.float32))
def to_dataset(x, y, x_transform=to_float32, y_transform = to_float32):
    assert len(x) == len(y)
    class ds(Dataset):
        def __init__(self):
            pass
        def __len__(self):
            return len(x)

        def __getitem__(self, idx):

            return x_transform(x[idx]), y_transform(y[idx])

    return ds()
