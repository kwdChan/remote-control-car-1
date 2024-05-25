
from multiprocessing.managers import BaseManager, BaseProxy, SyncManager, ValueProxy
from typing import Optional, Tuple, TypeVar, Union, List, cast, Dict
from typing_extensions import deprecated


from components import Component, shared_value
from data_collection.data_collection import Logger, LoggerSet
import numpy as np 
import time

class BlueToothCarControlSPP(Component):
    def __init__(self, logger: Logger):
        self.logger = logger
        

        # states
        self.speed = 50
        self.angular_speed = 50

    def step(self, data={})->Tuple[float, float]:
        self.logger.log_time()
        self.logger.log('keys', data)

        if (up:=data.get('up')) and up:
            self.speed += 0.5
            

        if (down:=data.get('down')) and down:
            self.speed -= 0.5

        if (cross:=data.get('cross')) and cross:
            self.angular_speed = 0

        if (cir:=data.get('cir')) and cir:
            self.angular_speed += 0.5

        if (sq:=data.get('sq')) and sq:
            self.angular_speed -= 0.5

        angular_velocity_mul = 0
        if (left:=data.get('left')) and left:

            angular_velocity_mul += 1

        if (right:=data.get('right')) and right:
            angular_velocity_mul -= 1 


        angular_velocity = max(0, self.angular_speed)*angular_velocity_mul

        if (go:=data.get('go')) and go:
            speed = self.speed
        else:
            speed = 0
        
        
        self.logger.log('angular_velocity', angular_velocity)
        self.logger.log('speed', speed)

        return angular_velocity, speed

    @classmethod
    def create_shared_outputs(cls, manager:BaseManager)->List[Optional[BaseProxy]]:
        assert isinstance(manager, SyncManager)
        angular_velocity = manager.Value('d', 0)
        speed = manager.Value('d', 0)
        return [angular_velocity, speed]


    @classmethod
    def create_shared_outputs_rw(cls, manager:BaseManager):
        assert isinstance(manager, SyncManager)

        angular_velocity_r, angular_velocity_w = shared_value(manager, 'd', 0)
        speed_r, speed_w = shared_value(manager, 'd', 0)

        return [angular_velocity_r, speed_r], [angular_velocity_w, speed_w]


    @classmethod
    def entry(
        cls, 
        logger_set:Optional[LoggerSet]=None, 
        **kwargs
    ):

        assert logger_set, "logger_set cannot be left empty"

        logger = logger_set.get_logger(**kwargs)
        return cls(
            logger, 
        )

