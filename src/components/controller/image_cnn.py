from typing import Optional, List
from components import Component, default_proxy_reader
from pathlib import Path
import numpy as np 
from multiprocessing.managers import BaseManager, BaseProxy, SyncManager, ValueProxy

import tensorflow as tf
import tensorflow.keras as keras # type: ignore 
from data_collection.data_collection import LoggerSet, Logger

class ImageMLController(Component):

    def __init__(self, model_path: Path, logger: Logger):
        model = model = keras.models.load_model(model_path) 

        self.model_path = model_path
        self.model = model
        self.logger = logger

    
    def step(self, arr:Optional[np.ndarray] = None): 
        assert not (arr is None)

        out = self.model(arr[None, :])[0]

        return out[0], out[1]#, out[2], out[3]

    @classmethod
    def create_shared_outputs(cls, manager: BaseManager) -> List[Optional[BaseProxy]]:
        """
        override this method to set the ctypes and initial values for the shared values 
        use the type hint to infer by default 
        """ 
        
        assert isinstance(manager, SyncManager)
        out0 = manager.Value('d', 0)
        out1 = manager.Value('d', 0)

        return [out0, out1]


    @classmethod
    def entry(
        cls, model_path:str='', 
        logger_set: Optional[LoggerSet]=None, 
        **kwargs
        ):

        assert isinstance(logger_set, LoggerSet)
        logger = logger_set.get_logger(**kwargs)

        return cls(Path(model_path), logger)

    
