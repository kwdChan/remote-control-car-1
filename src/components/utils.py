from typing import List, Dict, Literal, Union, Tuple, Any
from multiprocessing.connection import Connection
from data_collection.data_collection import  Logger


def receive_latest(receiver: Connection, logger: Logger, default=None):
    result = None
    while receiver.poll():
        result = receiver.recv()
    
    if not (result is None):
        values, sender_name, sender_idx = result
        logger.log('sender_name', sender_name)
        logger.log('sender_idx', sender_idx)
        return values
    else:
        return default 


def send(value, sender: Connection, logger: Logger):
    sender.send((value, logger.name, logger.idx))