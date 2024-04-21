from abc import abstractmethod
import time
from types import FunctionType
from typing import Callable, Tuple, TypeVar, get_origin, get_args, Union, Any
from typing_extensions import List, overload, override
from multiprocessing.sharedctypes import Synchronized as SharedValue
from multiprocessing.sharedctypes import SynchronizedArray as SharedArray
from multiprocessing import Value, Process

from multiprocessing import Manager
from multiprocessing.managers import BaseProxy, BaseManager
import warnings
import inspect
from ctypes import c_long, c_double, c_bool, c_wchar_p
import numpy as np
from data_collection.data_collection import Logger

def assign(variable: Union[None, BaseProxy], value):
    if variable is None: 
        return 

    elif hasattr(variable, "__setitem__"):
        variable[:] = value #type: ignore

    elif hasattr(variable, "value"):
        variable.value = value #type: ignore

    else: 
        raise NotImplementedError("no known value assignment method for this proxy")

class Component: 

    logger: Logger
    SHARED_VARIABLE_LIST_NONE_OKAY = List[Union[None, BaseProxy]]
    SHARED_VARIABLE_LIST_NOT_NONE = List[BaseProxy]

    @abstractmethod
    def step(self) -> Any:
        return ()

    @classmethod
    def entry(cls, **kwargs):
        return cls(**kwargs)
    
    @classmethod
    def main(cls, interval, past_due_warning_sec=np.inf, entry_kwargs={}, shared_inputs: SHARED_VARIABLE_LIST_NOT_NONE=[], shared_outputs: SHARED_VARIABLE_LIST_NONE_OKAY=[]):
        """
        main loop: 
        object instantiate (through cls.entry) and then loop 
        """
        
        obj = cls.entry(**entry_kwargs)

        t_last = time.monotonic()
        while True: 
            now = time.monotonic()

            time_passed = now - t_last
            time_past_due = time_passed - interval
            if time_past_due >= 0: 
                if time_past_due > past_due_warning_sec:
                    warnings.warn(f"time_past_due: {time_past_due}, interval: {interval}")
                t_last = now
                obj.logger.increment_idx()

                outputs = obj.step(*[v._getvalue() for v in shared_inputs])
                if outputs is None: 
                    outputs = ()
                for idx, o in enumerate(outputs): 
                    assign(shared_outputs[idx], o)
            else: 
                time.sleep(interval/50)


    # INFERRED_TYPES = {
    #     int: (c_long, 0),
    #     float: (c_double, 0),
    #     #str: (c_wchar_p, ""),
    #     bool: (c_bool, False)
    # }

    @classmethod
    def create_shared_outputs(cls, manager: BaseManager) -> SHARED_VARIABLE_LIST_NONE_OKAY:
        """
        override this method to set the ctypes and initial values for the shared values 
        use the type hint to infer by default 
        """ 
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
    def create(cls, manager: BaseManager, **main_kwargs) -> Tuple[SHARED_VARIABLE_LIST_NONE_OKAY, "function"]:
        """
        create the output and then 
        """
        
        shared_outputs = cls.create_shared_outputs(manager)

        def starter(shared_inputs: List[SharedValue]=[]) -> Process:
            process = Process(
                target=cls.main, 
                kwargs=dict(
                    shared_inputs=shared_inputs,
                    shared_outputs=shared_outputs, 
                    **main_kwargs
                    )
                )
            process.start()

            return process

        return shared_outputs, starter


        

