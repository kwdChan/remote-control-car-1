from typing import Any, Callable, List, cast
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, BatchSampler
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from IPython.display import display

def num_param(model: nn.Module):
    i = 0
    for k, v in model.state_dict().items():
        i += np.prod(v.shape)
    return i


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
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            eval_loss += loss_fn(y_pred, y).item()

        # TODO: metrics
        
    eval_loss /= num_batches # TODO: this is not right

    if not silent: 
        print(f'eval_loss: {eval_loss}')
    return cast(float, eval_loss)
    
    
def evaluate_v2(model: nn.Module, dataloader: DataLoader):
    device = next(model.parameters()).device
    model.eval()

    y_preds = []
    ys = []

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            y_preds.append(y_pred)
            ys.append(y)

        
    return torch.concat(y_preds), torch.concat(ys)

def check_with_patient(n_step):
    current_min = np.inf
    n_step_left = n_step
    def out_of_patient(new_value):
        nonlocal current_min
        nonlocal n_step_left

        if current_min <= new_value:
            n_step_left -= 1 
    
        else:
            current_min = new_value
            n_step_left = n_step

        return n_step_left <= 0
    return out_of_patient


from torch.optim.lr_scheduler import MultiplicativeLR
def find_lr(
        model_instantiator: Callable[[],  nn.Module], 
        train_dataloader: DataLoader, 
        optimiser_class: type[optim.Optimizer], 
        loss_fn, 
        device='cuda', 
        starting_lr=1e-7, 
        factor=10**(1/20), 
        epochs=100, 
        silent=True, 
        patient=10, 
    ):
    out_of_patient = check_with_patient(patient)

    fig = UpdatingPlotlyLines('epoch', ['train_loss', 'lr'])
    fig2 = UpdatingPlotlyLines('lr', ['train_loss'])
    model = model_instantiator().to(device)
    optimiser = optimiser_class(model.parameters(), lr=torch.tensor(starting_lr)) #type: ignore

    lr_schr = MultiplicativeLR(optimiser, lambda e: factor)

    fig.display()
    fig2.display()
    
    for epoch in range(epochs):

        train_loss = fit_step(model, train_dataloader, optimiser, loss_fn, silent=silent)

        lr = factor**(epoch-1)*starting_lr
        fig.append(epoch=epoch, train_loss=train_loss, lr=lr)
        fig2.append(lr=lr, train_loss=train_loss)
        lr_schr.step()

        if out_of_patient(train_loss):
            return 



def fit(
        model: nn.Module, 
        train_dataloader: DataLoader, 
        val_dataloader:DataLoader, 
        optimiser: optim.Optimizer, 
        loss_fn, 
        epochs: int, 
        silent=True
    ):
    """
    reference implementation 
    """

    fig = UpdatingPlotlyLines('epoch', ['train_loss', 'val_loss'])
    fig.display()
    for epoch in range(epochs):
        train_loss = fit_step(model, train_dataloader, optimiser, loss_fn, silent=silent)
        val_loss = evaluate(model, val_dataloader, loss_fn, silent=silent)

        fig.append(epoch=epoch, train_loss=train_loss, val_loss=val_loss)

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
