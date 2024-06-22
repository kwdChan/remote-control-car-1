from typing import Callable, Optional, List, Any, Tuple

from components import ComponentInterface, CallChannel, component, sampler, samples_producer, rpc, declare_method_handler, loop
from components.logger import LoggerComponent

from pathlib import Path
import numpy as np 
from multiprocessing.managers import BaseManager, BaseProxy, SyncManager, ValueProxy
import time
import tensorflow as tf
import tensorflow.keras as keras # type: ignore 
from data_collection.data_collection import LoggerSet, Logger
@component
class ImageMLControllerV3b(ComponentInterface):

    def __init__(self, func: Callable[[], Callable[[np.ndarray], Tuple[int, int]]], log, log_time, increment_index, name):

        self.model = func()

        self.log = declare_method_handler(log, LoggerComponent.log)
        self.log_time = declare_method_handler(log_time, LoggerComponent.log_time)
        self.increment_index = declare_method_handler(increment_index, LoggerComponent.increment_index)

        self.name = name

    
    # TODO: add time data and ignore the frames if the handler isn't fast enough
    @rpc()
    @sampler
    def step(self, arr:np.ndarray): 

        self.increment_index.call_no_return(self.name)
        t0 = time.monotonic()

        self.log_time.call_no_return(self.name, 'before')

        assert not (arr is None)

        v0, v1 = self.model(arr)
        t1 = time.monotonic()

        self.log.call_no_return(self.name, dict(timelapsed=t1-t0))
        
        return v0, v1



@component
class ImageMLControllerV3a(ComponentInterface):

    def __init__(self, func: Callable[[], Callable[[np.ndarray], Tuple[int, int]]], log, log_time, increment_index, name):

        self.model = func()

        self.log = declare_method_handler(log, LoggerComponent.log)
        self.log_time = declare_method_handler(log_time, LoggerComponent.log_time)
        self.increment_index = declare_method_handler(increment_index, LoggerComponent.increment_index)

        self.name = name

    
    # TODO: add time data and ignore the frames if the handler isn't fast enough
    @loop
    @sampler
    def step(self, arr): 

        self.increment_index.call_no_return(self.name)
        t0 = time.monotonic()

        self.log_time.call_no_return(self.name, 'before')

        assert not (arr is None)

        v0, v1 = self.model(arr)
        t1 = time.monotonic()

        self.log.call_no_return(self.name, dict(timelapsed=t1-t0))
        
        return v0, v1



# class ImageMLControllerV2(Component):

#     def __init__(self, func: Callable[[], Callable[[np.ndarray], Tuple[int, int]]], logger: Logger):

#         self.model = func()
#         self.logger = logger

    
#     def step(self, arr:Optional[np.ndarray] = None): 
#         t0 = time.monotonic()
#         self.logger.log_time('before')
#         assert not (arr is None)

#         v0, v1 = self.model(arr)
#         t1 = time.monotonic()
#         self.logger.log('timelapsed', t1-t0)
#         return v0, v1

#     @classmethod
#     def create_shared_outputs_rw(cls, manager: BaseManager):
#         """
#         override this method to set the ctypes and initial values for the shared values 
#         use the type hint to infer by default 
#         """ 
        
#         assert isinstance(manager, SyncManager)
#         out0r, out0w = shared_value(manager, 'd', 0)
#         out1r, out1w = shared_value(manager, 'd', 0)

#         return [out0r, out1r], [out0w, out1w]

#     @classmethod
#     def entry(
#         cls, func:Any=None, 
#         logger_set: Optional[LoggerSet]=None, 
#         **kwargs
#         ):

#         assert isinstance(logger_set, LoggerSet)
#         logger = logger_set.get_logger(**kwargs)

#         return cls(func, logger)

    


# class ImageMLController(Component):

#     def __init__(self, model_path: Path, logger: Logger):
#         model = keras.models.load_model(model_path) 

#         self.model_path = model_path
#         self.model = model
#         self.logger = logger

    
#     def step(self, arr:Optional[np.ndarray] = None): 
#         t0 = time.monotonic()
#         self.logger.log_time('before')
#         assert not (arr is None)
#         out = self.model(arr[None, :])[0]
#         t1 = time.monotonic()
#         self.logger.log('timelapsed', t1-t0)
#         return out[0], out[1]#, out[2], out[3]

#     @classmethod
#     def create_shared_outputs_rw(cls, manager: BaseManager):
#         """
#         override this method to set the ctypes and initial values for the shared values 
#         use the type hint to infer by default 
#         """ 
        
#         assert isinstance(manager, SyncManager)
#         out0r, out0w = shared_value(manager, 'd', 0)
#         out1r, out1w = shared_value(manager, 'd', 0)

#         return [out0r, out1r], [out0w, out1w], 

#     @classmethod
#     def entry(
#         cls, model_path:str='', 
#         logger_set: Optional[LoggerSet]=None, 
#         **kwargs
#         ):

#         assert isinstance(logger_set, LoggerSet)
#         logger = logger_set.get_logger(**kwargs)

#         return cls(Path(model_path), logger)

    
