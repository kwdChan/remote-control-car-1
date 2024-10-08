from components.syncronisation import ComponentInterface, CallChannel, component, sampler, samples_producer, rpc, declare_method_handler
import numpy as np
from typing import Dict, List, Tuple, cast
from data_collection.data_collection import Logger, LoggerSet
import datetime


def add_time(data:Dict, key='time'):
    data = data.copy()
    assert not key in data
    data[key] = datetime.datetime.now()
    return data


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
    def log(self, name:str, data:Dict, idx:int):

        logger = self.get_create_logger(name)

        for k, v in data.items():
            logger.log(k, v, idx)

    @rpc()
    def setup_video_saver(self, name:str, resolution:Tuple[int, int], **kwargs):
        logger = self.get_create_logger(name)

        logger.setup_video_saver(resolution=resolution, **kwargs)

    @rpc()
    def save_video_frame(self, name:str, frame:np.ndarray, idx:int):
        logger = self.get_create_logger(name)

        try: 
            logger.save_video_frame(frame, idx)
        except:
            #TODO: should be the setup and save methods in one method
            self.setup_video_saver(name, resolution=cast(Tuple, frame.shape[:2]), framerate=30)
            logger = self.get_create_logger(name)
            logger.save_video_frame(frame, idx)

    @rpc()
    def save_logger(self, name: str):
        self.loggers[name].save()

    @rpc()
    def get_logger(self, name: str):
        return self.loggers[name]