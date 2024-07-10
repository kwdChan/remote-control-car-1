from typing import Callable, Optional, List, Any, Tuple, cast

from components.syncronisation import ComponentInterface, CallChannel, component, sampler, samples_producer, rpc, declare_method_handler, loop, ThreadHandler
from components.logger import LoggerComponent, add_time

from pathlib import Path
import numpy as np 
from multiprocessing.managers import BaseManager, BaseProxy, SyncManager, ValueProxy
import time
#import tensorflow as tf
#import tensorflow.keras as keras # type: ignore 
from data_collection.data_collection import LoggerSet, Logger
from utils import Timer
#from concurrent.futures import ThreadPoolExecutor


# def get_model():
#     model_path = './2Jun-pi.keras'
#     ml_model = keras.models.load_model(model_path) 
#     def model(arr):

#         out = ml_model(arr[None, :]/255)[0]
#         out *= 100
#         out = out.numpy()

#         return out[0], out[3]
#     return model

# def get_model2():
#     model_path = './2Jun-pi.keras'
#     ml_model = keras.models.load_model(model_path) 
#     def model(arr, values: List):
        
#         values = cast(np.ndarray, np.stack(values, axis=0)).mean(0) 

#         out = ml_model(arr[None, :]/255, values)[0]
#         out *= 100
#         out = out.numpy()

#         return out[0], out[3]
#     return model

@component
class FrameMLToAngularVelocity(ComponentInterface):

    ModelFuncType = Callable[[np.ndarray], float]

    def __init__(self, func: Callable[[], ModelFuncType], log, name, delay_ms):

        # objects
        self.model = func()
        self.log = declare_method_handler(log, LoggerComponent.log)
        self.handler = ThreadHandler()

        # constants
        self.name = name
        self.delay_ms = delay_ms

        # states
        self.idx = 0
        self.out = 0


    @rpc()
    @sampler
    def run_model(self, arr:np.ndarray): 
        with Timer() as timer: 
            self.out = self.model(arr)

        data = {}
        data['timelapsed'] = timer.timelapsed
        data['out'] = self.out

        self.handler.call(self.output, time_wait_ms=max(0, self.delay_ms-timer.timelapsed*1000))

        self.log.call_no_return(self.name, add_time(data, 'run_model'), self.idx)
        self.idx += 1

    @samples_producer(typecodes=['d'], default_values=[0]) 
    def output(self, time_wait_ms: float):
        time.sleep(time_wait_ms/1000)
        return (self.out, )



@component
class ImageMLControllerV4(ComponentInterface):

    ModelFuncType = Callable[[np.ndarray, List[Any]], Tuple[float, float]]

    def __init__(self, func: Callable[[], ModelFuncType], log, name):

        self.model = func()

        self.log = declare_method_handler(log, LoggerComponent.log)

        self.handler = ThreadHandler()

        self.name = name
    
        self.idx = 0

        self.scalars = []

        self.v0 = 0
        self.v1 = 0


    @rpc()
    def pass_values(self, value):
        self.scalars.append(value)

    def get_reset_values(self):
        scalars, self.scalars = self.scalars, []
        return scalars

    @rpc()
    @sampler
    def run_model(self, arr:np.ndarray): 
        with Timer() as timer: 
            scalars = self.get_reset_values()
            self.v0, self.v1 = self.model(arr, scalars)

        data = {}
        data['timelapsed'] = timer.timelapsed
        data['v0'] = self.v0
        data['v1'] = self.v1

        self.handler.call(self.output, time_wait_ms=max(0, 100-timer.timelapsed*1000))

        self.log.call_no_return(self.name, add_time(data, 'run_model'), self.idx)
        self.idx += 1

    @samples_producer(typecodes=['d', 'd'], default_values=[0, 0]) 
    def output(self, time_wait_ms: float):
        time.sleep(time_wait_ms/1000)
        return self.v0, self.v1



@component
class ImageMLControllerV3b(ComponentInterface):

    def __init__(self, func: Callable[[], Callable[[np.ndarray], Tuple[int, int]]], log, name):

        self.model = func()

        self.log = declare_method_handler(log, LoggerComponent.log)

        self.name = name

        self.idx = 0

    
    @rpc()
    @sampler
    @samples_producer(typecodes=['d', 'd'], default_values=[0, 0])
    def step(self, arr:np.ndarray): 

        t0 = time.monotonic()
        data = {}

        data = add_time(data, 'before')

        assert not (arr is None)

        v0, v1 = self.model(arr)
        t1 = time.monotonic()

        data['timelapsed'] = t1-t0

        self.log.call_no_return(self.name, add_time(data), self.idx)
        self.idx += 1
        
        return v0, v1



# @component
# class ImageMLControllerV3a(ComponentInterface):

#     def __init__(self, func: Callable[[], Callable[[np.ndarray], Tuple[int, int]]], log, log_time, increment_index, name):

#         self.model = func()

#         self.log = declare_method_handler(log, LoggerComponent.log)
#         self.log_time = declare_method_handler(log_time, LoggerComponent.log_time)
#         self.increment_index = declare_method_handler(increment_index, LoggerComponent.increment_index)

#         self.name = name

    
#     # TODO: add time data and ignore the frames if the handler isn't fast enough
#     @loop
#     @sampler
#     def step(self, arr): 

#         self.increment_index.call_no_return(self.name)
#         t0 = time.monotonic()

#         self.log_time.call_no_return(self.name, 'before')

#         assert not (arr is None)

#         v0, v1 = self.model(arr)
#         t1 = time.monotonic()

#         self.log.call_no_return(self.name, dict(timelapsed=t1-t0))
        
#         return v0, v1



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

    
