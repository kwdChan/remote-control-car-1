from components import ComponentInterface, CallChannel, component, sampler, samples_producer, rpc, declare_method_handler
import numpy as np
from typing import Dict, List, Tuple
from data_collection.data_collection import Logger, LoggerSet


@component
class LoggerComponent(ComponentInterface):
    """
    this is written in an awkward way to preserve the old data structure...
    """

    def __init__(self, loggerset: LoggerSet):

        self.loggerset = loggerset

        self.loggers: Dict[str, Logger] = {}


    def get_create_logger(self, name):
        if not name in self.loggers:
            self.loggers[name] = self.loggerset.get_logger(name, save_interval=15)
        return self.loggers[name]

    @rpc()
    def increment_index(self, name:str):

        logger = self.get_create_logger(name)

        logger.increment_idx()
        logger.log_time()

    @rpc() 
    def log(self, name:str, data:Dict):

        logger = self.get_create_logger(name)

        for k, v in data.items():
            logger.log(k, v)

    @rpc()
    def log_time(self, name:str, key:str):

        logger = self.get_create_logger(name)

        logger.log_time(key)

    @rpc()
    def setup_video_saver(self, name:str, resolution:Tuple[int, int], **kwargs):
        logger = self.get_create_logger(name)

        logger.setup_video_saver(resolution=resolution, **kwargs)

    @rpc()
    def save_video_frame(self, name:str, frame:np.ndarray):
        logger = self.get_create_logger(name)
        logger.save_video_frame(frame)

        
