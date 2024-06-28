from typing import Optional, Tuple, TypeVar, Union, List, cast, Dict

from components.syncronisation import ComponentInterface, CallChannel, component, sampler, samples_producer, rpc, declare_method_handler, loop
from components.logger import LoggerComponent, add_time

import numpy as np 

# work with bluetooth server v2, not v3
@component
class BlueToothCarControlSPP_V2(ComponentInterface):
    def __init__(self, log, name='BlueToothCarControlSPP_V2'):

        self.log = declare_method_handler(log, LoggerComponent.log)
        

        self.name = name
        self.up_reactor = get_press_reaction(self.up_handler)
        self.down_reactor = get_press_reaction(self.down_handler)
        self.cir_reactor = get_press_reaction(self.cir_handler)
        self.sq_reactor = get_press_reaction(self.sq_handler)


        # states
        self.speed = 50
        self.angular_speed = 135
        self.boost = False

        self.idx = 0

    def up_handler(self):
        self.speed += 5
    
    def down_handler(self):
        self.speed -= 5
    
    def cir_handler(self):
        self.angular_speed += 22.5
    
    def sq_handler(self):
        self.angular_speed -= 22.5

    @loop
    @samples_producer(typecodes =['d', 'd'], default_values=[0, 0])
    @sampler
    def step(self, data={})->Tuple[float, float]:

        
        log_data = {}
        log_data['keys'] = data

        self.up_reactor(data.get('up'))
        self.down_reactor(data.get('down'))
        self.cir_reactor(data.get('cir'))
        self.sq_reactor(data.get('sq'))

        if data.get('cross'):
            self.boost = True


        angular_velocity_mul = 0
        if data.get('left'):
            self.boost = False
            angular_velocity_mul += 1

        if data.get('right'):
            self.boost = False
            angular_velocity_mul -= 1 

        angular_velocity = max(0, self.angular_speed)*angular_velocity_mul

        if data.get('go'):
            speed = 200 if self.boost else self.speed 
        else:
            speed = 0
            self.boost = False
        
        log_data['angular_velocity'] = angular_velocity
        log_data['speed'] = speed

        self.log.call_no_return(self.name, add_time(log_data), self.idx)
        self.idx += 1
        return angular_velocity, speed


def get_press_reaction(handler):

    already_up = False
    def check(isup, *arg, **kwargs):
        nonlocal already_up
        
        if isup and not already_up:
            already_up = True
            handler(*arg, **kwargs)
        if not isup:
            already_up = False
    return check 
    

# class BlueToothCarControlSPP(Component):
#     def __init__(self, logger: Logger):
#         self.logger = logger
        


#         self.up_reactor = get_press_reaction(self.up_handler)
#         self.down_reactor = get_press_reaction(self.down_handler)
#         self.cir_reactor = get_press_reaction(self.cir_handler)
#         self.sq_reactor = get_press_reaction(self.sq_handler)


#         # states
#         self.speed = 50
#         self.angular_speed = 135
#         self.boost = False

#     def up_handler(self):
#         self.speed += 5
    
#     def down_handler(self):
#         self.speed -= 5
    
#     def cir_handler(self):
#         self.angular_speed += 22.5
    
#     def sq_handler(self):
#         self.angular_speed -= 22.5


        
#     def step(self, data={})->Tuple[float, float]:
#         self.logger.log_time()
#         self.logger.log('keys', data)

#         self.up_reactor(data.get('up'))
#         self.down_reactor(data.get('down'))
#         self.cir_reactor(data.get('cir'))
#         self.sq_reactor(data.get('sq'))

#         if data.get('cross'):
#             self.boost = True


#         angular_velocity_mul = 0
#         if data.get('left'):
#             self.boost = False
#             angular_velocity_mul += 1

#         if data.get('right'):
#             self.boost = False
#             angular_velocity_mul -= 1 

#         angular_velocity = max(0, self.angular_speed)*angular_velocity_mul

#         if data.get('go'):
#             speed = 200 if self.boost else self.speed 
#         else:
#             speed = 0
#             self.boost = False
        
        
#         self.logger.log('angular_velocity', angular_velocity)
#         self.logger.log('speed', speed)

#         return angular_velocity, speed

#     @classmethod
#     def create_shared_outputs(cls, manager:BaseManager)->List[Optional[BaseProxy]]:
#         assert isinstance(manager, SyncManager)
#         angular_velocity = manager.Value('d', 0)
#         speed = manager.Value('d', 0)
#         return [angular_velocity, speed]


#     @classmethod
#     def create_shared_outputs_rw(cls, manager:BaseManager):
#         assert isinstance(manager, SyncManager)

#         angular_velocity_r, angular_velocity_w = shared_value(manager, 'd', 0)
#         speed_r, speed_w = shared_value(manager, 'd', 0)

#         return [angular_velocity_r, speed_r], [angular_velocity_w, speed_w]


#     @classmethod
#     def entry(
#         cls, 
#         logger_set:Optional[LoggerSet]=None, 
#         **kwargs
#     ):

#         assert logger_set, "logger_set cannot be left empty"

#         logger = logger_set.get_logger(**kwargs)
#         return cls(
#             logger, 
#         )

