from abc import abstractmethod
import time
from typing import Callable, Tuple, TypeVar, get_origin, get_args, Union, Any, Optional, Dict, List

from multiprocessing import Value, Process

from multiprocessing import Manager, Value, Array
from multiprocessing.managers import BaseProxy, BaseManager, SyncManager

from typing_extensions import deprecated
import warnings
import inspect
import numpy as np
from data_collection.data_collection import Logger
from typing import Type
import array

T = TypeVar('T')
def get_switch(v1: Callable[[], T], v2: Callable[[], T], use1: Callable[[], bool]) -> Callable[[], T]:

    return lambda: v1() if use1() else v2()


def get_switches(v1: List[Callable[[], T]], v2: List[Callable[[], T]], use1: Callable[[], bool]) -> List[Callable[[], T]]:
    
    values = []
    for _v1, _v2 in zip(v1, v2):
        values.append(
            get_switch(_v1, _v2, use1 )
        )
    return values

def default_loop_v2( 
    instantiater, 
    init_kwargs, 
    interval,
    past_due_warning_sec=np.inf, 
    input_readers:List[Callable] = [], 
    output_assigners:List[Callable] = [],
    other_io:Dict[str, Callable] = {},     
):
    obj = instantiater(**init_kwargs, **other_io)

    t_last = time.monotonic()
    while True: 
        now = time.monotonic()

        time_passed = now - t_last
        time_past_due = time_passed - interval
        if time_past_due >= 0: 
            if time_past_due > past_due_warning_sec:
                warnings.warn(f"time_past_due: {time_past_due}, interval: {interval}")
            t_last = now
            
            outputs = obj.step(*[f() for f in input_readers])
            if outputs is None: 
                outputs = ()
            for idx, o in enumerate(outputs): 
                output_assigners[idx](o)

            obj.logger.increment_idx() 

        else: 
            time.sleep(interval/50)


def shared_list(typecodes, default_values):
    """TODO: testing needed"""
    v_list = [Value(tc, dv) for tc, dv in zip(typecodes, default_values)]

    def reader():
        return [v.value for v in v_list]

    def assigner(slicer, value):
        v_list[slicer] = value

    return reader, assigner


def shared_value_v2(typecode: Any, default_value:Any=0) -> Tuple[Callable, Callable]:

    v = Value(typecode, default_value)
    def reader():
        return v.value

    def assigner(value):
        v.value=value

    return reader, assigner


@deprecated("use v2 (no manager)")
def shared_value(manager: SyncManager, typecode: Any, default_value:Any=0) -> Tuple[Callable, Callable]:
    proxy = manager.Value(typecode, default_value)
    def reader():
        return proxy.value

    def assigner(value):
        proxy.value=value

    return reader, assigner

def shared_np_array_v2(typecode: Any, default_value: np.ndarray) -> Tuple[Callable, Callable]:

    flatten_arr = Array(typecode, default_value.ravel().tolist())
    dim = default_value.shape

    def assigner(value: np.ndarray):
        flatten_arr[:] = array.array(typecode, value.ravel()) # type: ignore

    def reader():
        return np.array(flatten_arr[:]).reshape(dim)

    return reader, assigner


@deprecated("use v2 (no manager)")
def shared_np_array(manager: SyncManager, typecode: Any, default_value: np.ndarray) -> Tuple[Callable, Callable]:

    flatten_proxy = manager.Array(typecode, default_value.ravel().tolist())
    dim = default_value.shape

    def assigner(value: np.ndarray):
        flatten_proxy[:] = array.array(typecode, value.ravel()) # type: ignore

    def reader():
        return np.array(flatten_proxy[:]).reshape(dim)

    return reader, assigner




def default_component_process_starter_v3(
    target_class: Type["Component"], 
    init_kwargs: Dict, 
    mainloop: Callable, 
    main_kwargs: Dict, 
    manager: BaseManager, 
    shared_outputs_kwargs: Dict = {}, 
    instantiater: Optional[Callable] = None, 
) -> Tuple[List[Callable], Callable] :
    """
    create the output and a function that takes the input proxy to start the process

    main_kwargs: arguments to main except the proxies (
        i.e. shared_inputs, shared_outputs, shared_values
    )
    
    """
    
    output_readers, output_assigners = target_class.create_shared_outputs_rw(manager, **shared_outputs_kwargs)
    
    def starter(input_readers: List[BaseProxy]=[]) -> Process:
        process = Process(
            target=mainloop, 
            kwargs=dict(
                instantiater = instantiater if instantiater else target_class.entry, 
                init_kwargs = init_kwargs, 

                input_readers= input_readers, 
                output_assigners = output_assigners,
                **main_kwargs  
                )
            )
        process.start()
        return process

    return output_readers, starter


def default_component_process_starter_v2(
    target_class: Type["Component"], 
    init_kwargs: Dict, 
    mainloop: Callable, 
    main_kwargs: Dict, 
    manager: BaseManager, 
    shared_outputs_kwargs: Dict = {}
) -> Tuple[List[Callable], Callable] :
    """
    create the output and a function that takes the input proxy to start the process

    main_kwargs: arguments to main except the proxies (
        i.e. shared_inputs, shared_outputs, shared_values
    )
    
    """
    
    output_readers, output_assigners = target_class.create_shared_outputs_rw(manager, **shared_outputs_kwargs)
    
    def starter(input_readers: List[BaseProxy]=[]) -> Process:
        process = Process(
            target=mainloop, 
            kwargs=dict(
                instantiater = target_class.entry,  #bad! 
                init_kwargs = init_kwargs, 

                input_readers= input_readers, 
                output_assigners = output_assigners,
                **main_kwargs  
                )
            )
        process.start()
        return process

    return output_readers, starter



class Channel:
    def __init__(self, some_struction_specification, response_channel: "Channel"):
        """
        to be instantiated in the main 
        """
        self.shared_list = [] # some implementation of the shared list 
        # use a python list of shared values instead of array? 
        # how about char? 

        #request channel and response channel? 
        #request specifier? 
        self.response_channel = response_channel


    def poll(self):
        """need to both read the list and write the list"""
        msg_id, msg = self.shared_list.pop(0)
        return msg_id, msg 

    def request(self, msg_id, msg):

        self.shared_list.append(msg)

    def response(self, msg_id, msg):
        self.response_channel.request(msg, msg_id)




    
def server_loop(queue: Channel, handler):
    while True:
        req_id, req = queue.poll()
        res = handler(req)

        queue.response(req_id, res)






# make this into an interface 
class Component: 

    logger: Logger
    shared_values: List[BaseProxy]
    

    @abstractmethod
    def step(self) -> Tuple:
        # have to return tuple because the main will iterate through the output variables 
        return ()


    @classmethod
    def entry(cls, **kwargs):
        return cls(**kwargs)
        
    @classmethod
    def create_shared_outputs_rw(cls, manager: BaseManager)-> Tuple[List[Callable], List[Callable]]:
        return [], []

    @classmethod
    def create_shared_outputs(cls, manager: BaseManager) -> List[Optional[BaseProxy]]:
        """
        override this method to set the ctypes and initial values for the shared values 
        use the type hint to infer by default 
        """ 
        return [None]
        raise NotImplementedError("This method needs to be overriden")


        step_return_types = inspect.signature(cls.step).return_annotation
        
        if get_origin(step_return_types) is tuple:
            # e.g.: func() -> Tuple[str, int]
            output_types = get_args(step_return_types)
        else: 
            # e.g. func() -> str 
            output_types = (step_return_types, )

        shared_outputs = []
        for o in output_types:
            if not o in cls.INFERRED_TYPES:
                raise ValueError(f"The provided type:{o} is cannot be inferred. Override this method.")
            shared_outputs.append(Value(*cls.INFERRED_TYPES[o]))

        return shared_outputs