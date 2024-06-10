import requests 

from tg_token import TOKEN

import datetime
import tensorflow as tf
import tensorflow.keras as keras # type: ignore


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