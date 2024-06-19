from components import component, sampler, samples_producer, event_handler, rpc
from components import EventBroadcaster, ComponentInterface, MessageChannel, EventEnum
import numpy as np
from typing import Dict, List, Tuple
from data_collection.data_collection import Logger, LoggerSet



def increment_index_event(event_broadcaster: EventBroadcaster, name):
    """
    this implementation allow logger name collision. not very safe. 
    """
    event_broadcaster.publish(dict(event_type=EventEnum.increment_index, name=name))
    
def log_event(event_broadcaster: EventBroadcaster, name: str,  data:Dict,):
    event_broadcaster.publish(dict(event_type=EventEnum.log, data=data, name = name))
    
def log_time_event(event_broadcaster: EventBroadcaster,  name: str, key:str,):
    event_broadcaster.publish(dict(event_type=EventEnum.log_time, key=key, name = name))

def setup_video_saver_event(event_broadcaster: EventBroadcaster,  name: str, resolution:Tuple[int, int], **kwargs):
    kwargs['resolution'] = resolution
    event_broadcaster.publish(dict(event_type=EventEnum.setup_video_saver, name = name, kwargs=kwargs))


    


@component({})
class LoggerComponent(ComponentInterface):
    """
    this is written in an awkward way to preserve the old data structure...
    """

    def __init__(self, loggerset: LoggerSet):

        self.loggerset = loggerset


        self.loggers: Dict[str, Logger] = {}
        self.component_idx: Dict[str, int] = {}
    
    @event_handler
    def log_handler(self, msg):
        event_type = msg.get('event_type')

        if event_type == EventEnum.increment_index:
            self.increment_index(msg.get('name'))

        elif event_type == EventEnum.log:
            self.log(msg.get('data'), msg.get('name'))
        
        elif event_type == EventEnum.log_time:
            self.log_time(msg.get('key'), msg.get('name'))

        elif event_type == EventEnum.setup_video_saver:
            self.setup_video_saver(msg.get('name'), msg.get('kwargs'))

        elif event_type == EventEnum.video_frame:
            self.save_video_frame(msg.get('name'), msg.get('frame'))


    def get_create_logger(self, name):
        if not name in self.loggers:
            self.loggers[name] = self.loggerset.get_logger(name, save_interval=15)
        return self.loggers[name]

    def increment_index(self, name):

        logger = self.get_create_logger(name)

        logger.increment_idx()
        logger.log_time()
        
    def log(self, data:Dict, name):

        logger = self.get_create_logger(name)

        for k, v in data.items():
            logger.log(k, v)

    def log_time(self, key, name):

        logger = self.get_create_logger(name)

        logger.log_time(key)


    def setup_video_saver(self, name, kwargs):
        logger = self.get_create_logger(name)

        logger.setup_video_saver(**kwargs)

    def save_video_frame(self, name, frame):
        logger = self.get_create_logger(name)
        logger.save_video_frame(frame)

        
