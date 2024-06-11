import requests 

from tg_token import TOKEN

import datetime
import tensorflow as tf
import tensorflow.keras as keras # type: ignore
import numpy as np

def find_lr(model, x, y=None, patience=10, start_lr=1e-6, epochs=100, verbose=1, **kwargs):
    
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: start_lr*10**(epoch/20)
    )
    
    history = model.fit(
        x, y, 
        epochs=epochs, 
        verbose=verbose,
        callbacks=[
            lr_schedule, keras.callbacks.EarlyStopping(monitor='loss', patience=patience), 
            use_tensorboard('find_lr'), 
            end_epoch_notify(),
        ], 
    )
    return history

def use_tensorboard(key, main_dir='logs', append_time=True, histogram_freq=1, **kwargs):
    log_dir = f"{main_dir}/{key}" 
    if append_time: 
        log_dir += datetime.datetime.now().strftime("/%m-%d/%H:%M:%S")
    return keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq, **kwargs)

def tg_notify(msg): 
    requests.get(f"https://api.telegram.org/bot{TOKEN}/sendMessage", dict(chat_id='988152989', text=msg))
    
def pretty_dict(data):
    r = ""
    for k, v in data.items():
        r += f"{k}: {v}\n"
    return r[:-1]

def end_epoch_notify():
    return keras.callbacks.LambdaCallback(on_train_end=lambda log:tg_notify(pretty_dict(log)) )


def validation_chunk_split(n_sample, chuck_size_minmax=(10, 200), val_split=0.3):
    chunk_min, chunk_max = chuck_size_minmax
    assert chunk_min

    split_points = [0]
    idx = 0

    while idx <= n_sample: 
        size = np.random.randint(chunk_max+chunk_min)+chunk_min

        idx = size + idx 
        split_points.append(idx)

    split_points[-1] = n_sample 

    val_idx = []
    train_idx = []
    for _i in range(1, len(split_points)):
        
        
        indices = list(range(split_points[_i-1], split_points[_i]))

        if np.random.rand() > val_split:
            train_idx += indices
        else:
            val_idx += indices


    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    
    return train_idx, val_idx
