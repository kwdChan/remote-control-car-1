from abc import abstractmethod
import time
from types import FunctionType
from typing import Callable, Tuple, TypeVar, get_origin, get_args, Union, Any, Optional, Dict, List
from multiprocessing.sharedctypes import Synchronized as SharedValue
from multiprocessing.sharedctypes import SynchronizedArray as SharedArray
from multiprocessing import Value, Process

from multiprocessing import Manager
from multiprocessing.managers import BaseProxy, BaseManager, SyncManager
from typing_extensions import deprecated
import warnings
import inspect
from ctypes import c_long, c_double, c_bool, c_wchar_p
import numpy as np
from data_collection.data_collection import Logger
from typing import Type
import array


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

def shared_value(manager: SyncManager, typecode: Any, default_value:Any=0) -> Tuple[Callable, Callable]:
    proxy = manager.Value(typecode, default_value)
    def reader():
        return proxy.value

    def assigner(value):
        proxy.value=value

    return reader, assigner

def shared_np_array(manager: SyncManager, typecode: Any, default_value: np.ndarray) -> Tuple[Callable, Callable]:

    flatten_proxy = manager.Array(typecode, default_value.ravel().tolist())
    dim = default_value.shape

    def assigner(value: np.ndarray):
        flatten_proxy[:] = array.array(typecode, value.ravel()) # type: ignore

    def reader():
        return np.array(flatten_proxy[:]).reshape(dim)

    return reader, assigner


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



def default_loop( 
    instantiater, 
    init_kwargs, 
    proxy_assigner, 
    proxy_reader,
    interval,
    past_due_warning_sec=np.inf, 
    input_proxies=[], 
    output_proxies=[],
    other_proxies:Dict={},     
):
    obj = instantiater(**init_kwargs, **other_proxies)

    t_last = time.monotonic()
    while True: 
        now = time.monotonic()

        time_passed = now - t_last
        time_past_due = time_passed - interval
        if time_past_due >= 0: 
            if time_past_due > past_due_warning_sec:
                warnings.warn(f"time_past_due: {time_past_due}, interval: {interval}")
            t_last = now
            
            outputs = obj.step(*[proxy_reader(v) for v in input_proxies])
            if outputs is None: 
                outputs = ()
            for idx, o in enumerate(outputs): 
                proxy_assigner(output_proxies[idx], o)

            obj.logger.increment_idx() 

        else: 
            time.sleep(interval/50)



def default_proxy_assigner(variable: Union[None, BaseProxy], value):
    if variable is None: 
        return 

    elif hasattr(variable, "__setitem__"):
        variable[:] = value #type: ignore

    elif hasattr(variable, "value"):
        variable.value = value #type: ignore

    else: 
        raise NotImplementedError("no known value assignment method for this proxy")

def default_proxy_reader(variable: BaseProxy):

    if hasattr(variable, "value"):
        return variable.value #type: ignore
    else: 
        return variable._getvalue()


def default_component_process_starter(
    target_class: Type["Component"], 
    init_kwargs: Dict, 
    mainloop: Callable, 
    main_kwargs: Dict, 
    manager: BaseManager, 
    proxy_assigner: Callable = default_proxy_assigner, 
    proxy_reader: Callable = default_proxy_reader
) -> Tuple[List[Optional[BaseProxy]], Dict[str, BaseProxy], Callable] :
    """
    create the output and a function that takes the input proxy to start the process

    main_kwargs: arguments to main except the proxies (
        i.e. shared_inputs, shared_outputs, shared_values
    )
    
    """
    
    output_proxies = target_class.create_shared_outputs(manager)
    other_proxies = target_class.create_shared_values(manager)
    
    def starter(input_proxies: List[BaseProxy]=[], other_proxies: Dict[str, BaseProxy]={}) -> Process:
        process = Process(
            target=mainloop, 
            kwargs=dict(
                instantiater = target_class.entry,  #bad! 
                init_kwargs = init_kwargs, 
                proxy_assigner = proxy_assigner, 
                proxy_reader = proxy_reader,
                input_proxies = input_proxies, 
                output_proxies = output_proxies,
                other_proxies = other_proxies,   
                **main_kwargs  
                )
            )
        process.start()
        return process

    return output_proxies, other_proxies, starter






# make this into an interface 
class Component: 

    logger: Logger
    shared_values: List[BaseProxy]
    
    @staticmethod
    @deprecated('use create_component')
    def assign(variable: Union[None, BaseProxy], value):
        if variable is None: 
            return 

        elif hasattr(variable, "__setitem__"):
            variable[:] = value #type: ignore

        elif hasattr(variable, "value"):
            variable.value = value #type: ignore

        else: 
            raise NotImplementedError("no known value assignment method for this proxy")
    
    @staticmethod
    @deprecated('use create_component')
    def access(variable: BaseProxy):

        if hasattr(variable, "value"):
            return variable.value #type: ignore
        else: 
            return variable._getvalue()


    @abstractmethod
    def step(self) -> Tuple:
        # have to return tuple because the main will iterate through the output variables 
        return ()


    @classmethod
    def entry(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def create_component(
        cls, 
        init_kwargs: Dict, 
        mainloop: Callable, 
        main_kwargs: Dict, 
        manager: BaseManager, 
        proxy_assigner: Callable = default_proxy_assigner, 
        proxy_reader: Callable = default_proxy_reader
    ) -> Tuple[List[Optional[BaseProxy]], Dict[str, BaseProxy], "function"]:
        return default_component_process_starter(
            cls, 
            init_kwargs, 
            mainloop, 
            main_kwargs, 
            manager, 
            proxy_assigner, 
            proxy_reader
        )

    @classmethod
    @deprecated('use create_component')
    def main(
        cls, interval, past_due_warning_sec=np.inf, entry_kwargs={}, 
        shared_inputs: List[BaseProxy]=[], 
        shared_outputs: List[Optional[BaseProxy]]=[],
        shared_values: Dict[str, BaseProxy]={}
    ):
        """
        main loop: 
        object instantiate (through cls.entry) and then loop 
        """
        
        obj = cls.entry(**entry_kwargs, **shared_values)

        t_last = time.monotonic()
        while True: 
            now = time.monotonic()

            time_passed = now - t_last
            time_past_due = time_passed - interval
            if time_past_due >= 0: 
                if time_past_due > past_due_warning_sec:
                    warnings.warn(f"time_past_due: {time_past_due}, interval: {interval}")
                t_last = now
                

                outputs = obj.step(*[cls.access(v) for v in shared_inputs])
                if outputs is None: 
                    outputs = ()
                for idx, o in enumerate(outputs): 
                    cls.assign(shared_outputs[idx], o)

                obj.logger.increment_idx() 

                # for idx, o in enumerate(shared_outputs): 
                #     cls.assign(shared_outputs[idx], o)


            else: 
                time.sleep(interval/50)


    # INFERRED_TYPES = {
    #     int: (c_long, 0),
    #     float: (c_double, 0),
    #     #str: (c_wchar_p, ""),
    #     bool: (c_bool, False)
    # }

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

    @classmethod
    def create_shared_values(cls, manager: BaseManager) -> Dict[str, BaseProxy]:
        """
        these proxies are passed to the entry of the receiver AND SELF
        **through the shared_values parameter of the receiver's main**
        """
        return {}


    @classmethod
    @deprecated('use create_component')
    def create(cls, manager: BaseManager, **main_kwargs) -> Tuple[List[Optional[BaseProxy]], Dict[str, BaseProxy], "function"]:
        """
        create the output and a function that takes the input proxy to start the process

        main_kwargs: arguments to main except the proxies (
            i.e. shared_inputs, shared_outputs, shared_values
        )
        
        """

        
        shared_outputs = cls.create_shared_outputs(manager)
        shared_values = cls.create_shared_values(manager)
        
        def starter(shared_inputs: List[BaseProxy]=[], shared_values: Dict[str, BaseProxy]={}) -> Process:
            process = Process(
                target=cls.main, 
                kwargs=dict(
                    shared_inputs=shared_inputs,
                    shared_outputs=shared_outputs, 
                    shared_values=shared_values,
                    **main_kwargs
                    )
                )
            process.start()

            return process

        return shared_outputs, shared_values, starter


        

