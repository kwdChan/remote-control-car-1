from multiprocessing import Value
from typing import List, Dict, Literal, Union, Tuple, Any, TypeVar, cast, Generic
from multiprocessing.connection import Connection
from typing_extensions import deprecated
from data_collection.data_collection import Logger
import time
T = TypeVar('T')

class ReceiveLatest(Generic[T]):

    def __init__(self, receiver: Connection, logger: Logger, initial_value: T=None):
        self.value = initial_value 
        self.receiver = receiver
        self.logger = logger

    def get(self) -> T:
        self.value = cast(T, receive_latest(self.receiver, self.logger, default=self.value))
        return self.value

def receive_immediately(receiver: Connection, logger: Logger, default=None):
    result = None
    if receiver.poll():
        result = receiver.recv()
    
    if not (result is None):
        values, sender_name, sender_idx = result
        logger.log('sender_name', sender_name)
        logger.log('sender_idx', sender_idx)
        return values
    else:
        return default 

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


# from ctypes import Structure

# class SharedData():
#     def __init__(self, MyStructure: Structure, **initial_values):
#         self.shared = Value('data', MyStructure(**initial_values))
        
#     #def receiver

#     pass


import asyncio    
async def clear_data(recvr):
    """
    for notebook use
    """
    def clear():
        while recvr.poll():
            recvr.recv() 
    while True:
        clear()
        await asyncio.sleep(0.5)


class Connector:
    def __init__(self):
        self.connection_pairs = []

    def connect(self, receiver:  Connection, sender: Connection, ):
        self.connection_pairs.append((receiver, sender, ))

    def step(self):
        for receiver, sender in self.connection_pairs: 
            if receiver.poll():
                data = receiver.recv()
                #print(data)
                sender.send(data)
            else: 
                pass#print('no')

def loop_for_n_sec(loop, callback, n_sec):
    """
    for notebook use
    """    
    start_t = time.monotonic()
    while (time.monotonic() - start_t) < n_sec:
        loop()
    callback()