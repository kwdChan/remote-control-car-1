from enum import Enum
from components import component, sampler, samples_producer, event_handler, rpc
from components import EventBroadcaster, ComponentInterface, MessageChannel
import numpy as np
from typing import Dict, List
from data_collection.data_collection import Logger, LoggerSet

class EventType(Enum):
    increment_index = 1
    log = 2
    log_time = 3



def increment_index_event(event_broadcaster: EventBroadcaster, name):
    event_broadcaster.publish(dict(event_type=EventType.increment_index, name=name))
    
def log_event(event_broadcaster: EventBroadcaster, data:Dict, name: str):
    event_broadcaster.publish(dict(event_type=EventType.log, data=data, name = name))
    
def log_time_event(event_broadcaster: EventBroadcaster, key:str, name: str):

    event_broadcaster.publish(dict(event_type=EventType.log_time, key=key, name = name))
    


@component({})
class LoggerComponent(ComponentInterface):
    """
    this is written in an awkward way to preserve the old data structure...
    """

    def __init__(self, loggerset: LoggerSet):

        self.loggerset = loggerset


        self.loggers: Dict[str, Logger] = {}
        self.component_idx: Dict[str, int] = {}
    
    @event_handler("log")
    def log_handler(self, msg):
        event_type = msg.get('event_type')

        if event_type == EventType.increment_index:
            self.increment_index(msg.get('name'))

        elif event_type == EventType.log:
            self.log(msg.get('data'), msg.get('name'))
        
        elif event_type == EventType.log_time:
            self.log_time(msg.get('key'), msg.get('name'))



    def increment_index(self, name):

        logger = self.get_create_logger(name)

        logger.increment_idx()
        logger.log_time()
        
    def log(self, data, name):

        logger = self.get_create_logger(name)

        for k, v in data:
            logger.log(k, v)

    def log_time(self, key, name):

        logger = self.get_create_logger(name)

        logger.log_time(key)

    def get_create_logger(self, name):
        if not name in self.loggers:
            self.loggers[name] = self.loggerset.get_logger(name, save_interval=15)
        return self.loggers[name]

        

    

@component({"event_from_com1": None})
class MyTestComponent(ComponentInterface):
    def __init__(self, event_from_com1: EventBroadcaster):
        self.event_from_com1 = event_from_com1
        self.idx = -1

    @samples_producer(['d', 'd'], [0, np.zeros((4,4))])
    @sampler
    def step(self, idx_other, arr_other):
        self.idx += 1

        self.event_from_com1.publish({"step": self.idx})
        
        
        return self.idx, np.random.random((4,4))
        

    @event_handler("increment")
    def recv(self, msg):
        print(f'increment: {msg}')
        self.idx += msg


    @rpc()
    def com1_rpc(self, msg):

        print(f'com1 received msg: {msg}', flush=True)

        return msg

