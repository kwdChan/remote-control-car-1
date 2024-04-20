from abc import abstractmethod
import time
from types import FunctionType
from typing import Callable, Tuple, get_origin, get_args, Union, Any
from typing_extensions import List, overload, override
from multiprocessing.sharedctypes import Synchronized as SharedValue
from multiprocessing.sharedctypes import SynchronizedArray as SharedArray
from multiprocessing import Value, Process
import inspect
from ctypes import c_long, c_double, c_bool, c_wchar_p

from data_collection.data_collection import Logger
def access(v: Union[SharedValue, SharedArray]):
    return v[:] if isinstance(v, SharedArray) else v.value

def assign(variable: Union[None, SharedValue, SharedArray], value):
    if variable is None: 
        return 
    if isinstance(variable, SharedArray):
        variable[:] = value
    else: 
        variable.value  = value

class Component: 

    logger: Logger
    SHARED_VARIABLE_LIST = List[Union[None, SharedValue, SharedArray]]
    SHARED_VARIABLE_LIST_NOT_NONE = List[Union[SharedValue, SharedArray]]

    @abstractmethod
    def step(self) -> Any:
        return ()

    @classmethod
    def entry(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    



    @classmethod
    def main(cls, interval, entry_args=(), entry_kwargs={}, shared_inputs: SHARED_VARIABLE_LIST_NOT_NONE=[], shared_outputs: SHARED_VARIABLE_LIST=[]):
        """
        main loop: 
        object instantiate (through cls.entry) and then loop 
        """
        
        obj = cls.entry(*entry_args, **entry_kwargs)

        t_last = time.monotonic()
        while True: 
            now = time.monotonic()
            if (now - t_last) >= interval: 
                t_last = now
                obj.logger.increment_idx()
                outputs = obj.step(*[access(v) for v in shared_inputs])
            
                if outputs is None: 
                    outputs = ()
                for idx, o in enumerate(outputs): 
                    if not shared_outputs[idx] is None: 
                        assign(shared_outputs[idx], o)
            else: 
                time.sleep(interval/50)


    INFERRED_TYPES = {
        int: (c_long, 0),
        float: (c_double, 0),
        #str: (c_wchar_p, ""),
        bool: (c_bool, False)
    }


    @classmethod
    def create_shared_outputs(cls) -> SHARED_VARIABLE_LIST:
        """
        override this method to set the ctypes and initial values for the shared values 
        use the type hint to infer by default 
        """

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
    def create(cls, *main_args, **main_kwargs) -> Tuple[SHARED_VARIABLE_LIST, "function"]:
        """
        create the output and then 
        """
        
        shared_outputs = cls.create_shared_outputs()

        def starter(shared_inputs: List[SharedValue]=[]) -> Process:
            process = Process(
                target=cls.main, 
                args=main_args, 
                kwargs=dict(
                    shared_inputs=shared_inputs,
                    shared_outputs=shared_outputs, 
                    **main_kwargs
                    )
                )
            process.start()

            return process

        return shared_outputs, starter


        

