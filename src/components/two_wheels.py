from pickle import NONE
from typing import List, Dict, Literal, Union, Tuple, Any, Optional
import datetime, time
from typing_extensions import deprecated, override 
from components import Component
import numpy as np 
import pandas as pd
import RPi.GPIO as GPIO
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from data_collection.data_collection import LoggerSet, Logger
from dataclasses import dataclass

def setup_pwm(pin, freq=300) -> GPIO.PWM: #type: ignore
    GPIO.setup(pin, GPIO.OUT) #type: ignore
    return GPIO.PWM(pin, freq)  #type: ignore

class TwoWheelsV2(Component):
    def __init__(self, ch_left, ch_right, logger:Logger):
        ch_left.start(0)
        ch_right.start(0)
        self.ch_left = ch_left
        self.ch_right = ch_right
        self.logger = logger

    @override
    def step(
        self, 
        left: Optional[float]=None, 
        right: Optional[float]=None
    ) -> Tuple:
        
        if (left is None) or (right is None):
            print('nan')
            return ()

        left = min(100, max(0, left))
        right = min(100, max(0, right))
        
        self.ch_left.ChangeDutyCycle(left)
        self.ch_right.ChangeDutyCycle(right)

        self.logger.log_time('time')
        self.logger.log('left', left)
        self.logger.log('right', right)
        return ()

    @override
    @classmethod
    def entry(
        cls, 
        left_pin:int=0, 
        right_pin:int=0, 
        logger_set: Optional[LoggerSet]=None, 
        dir_pins: Optional[Tuple[int, int]] = None,
        name = '', 
        **kwargs
    ):
        """
        
        """
        assert name, "name cannot be left empty"
        assert left_pin, "left_pin cannot be left empty"
        assert right_pin, "right_pin cannot be left empty"
        assert logger_set, "logger_set cannot be left empty"
        assert dir_pins, "dir_pins cannot be left empty"

        ch_left = setup_pwm(left_pin, freq=300)
        ch_right = setup_pwm(right_pin, freq=300)
        ch_left.ChangeDutyCycle(0)
        ch_right.ChangeDutyCycle(0)

        GPIO.setup(dir_pins[0], GPIO.OUT) #type: ignore
        GPIO.setup(dir_pins[1], GPIO.OUT) #type: ignore
        GPIO.output(dir_pins[0], 0) #type: ignore
        GPIO.output(dir_pins[1], 0) #type: ignore

        logger = logger_set.get_logger(name, **kwargs)

        return cls(ch_left, ch_right, logger)

@deprecated('use V2')
class TwoWheels:
    def __init__(self, ch_left, ch_right, logger:Logger):
        ch_left.start(0)
        ch_right.start(0)
        self.ch_left = ch_left
        self.ch_right = ch_right
        self.logger = logger
 
    def step(self, left: Union[None, float], right: Union[None, float]) -> None:
        
        if (left is None) and (right is None):
            return 
        
        self.ch_left.ChangeDutyCycle(left)
        self.ch_right.ChangeDutyCycle(right)

        self.logger.log_time('time')
        self.logger.log('left', left)
        self.logger.log('right', right)

    @staticmethod    
    def sample(input_con: Connection) -> Any:
        """
        read only the latest input

        the component stale if the inputs are comming faster than it can read
        """
        result = None
        while input_con.poll():
            result = input_con.recv()
        return result


    @staticmethod
    def main(left_pin:int, right_pin:int, logger: Logger, input_con: Connection, dir_pins: Union[Tuple[int, int], None] = None):
        """
        
        """
        ch_left = setup_pwm(left_pin,freq=300)
        ch_right = setup_pwm(right_pin,freq=300)

        if not dir_pins is None:
            GPIO.setup(dir_pins[0], GPIO.OUT) #type: ignore
            GPIO.setup(dir_pins[1], GPIO.OUT) #type: ignore
            GPIO.output(dir_pins[0], 0) #type: ignore
            GPIO.output(dir_pins[1], 0) #type: ignore

        component = TwoWheels(ch_left, ch_right, logger)

        while True:
            logger.increment_idx()
        
            result = TwoWheels.sample(input_con)
            if not (result is None):
                (left, right), sender_name, sender_idx = result
                logger.log('sender_name', sender_name)
                logger.log('sender_idx', sender_idx)
                component.step(left, right)

            time.sleep(0.1)

    @staticmethod
    def start(left_pin, right_pin, logger_set: LoggerSet, dir_pins: Union[Tuple[int, int], None] = None, **kwargs):
        """
        using GPIO.BOARD pin
        """

        logger = logger_set.get_logger(**kwargs)
        receiver, sender = Pipe(False)        
        process = Process(target=TwoWheels.main, args=(left_pin, right_pin, logger, receiver, dir_pins))
        process.start()
        return process, sender


@deprecated("not used")
def speed_proportion_control(proportion, speed) -> Tuple[float, float, List]:
    """
    balance*speed = left speed
    (1-balance)*speed = right speed
    """
    warnings = []
    if not ((proportion >=0) and (proportion <= 1)):
        warnings.append({'proportion': proportion})
        proportion = min(proportion, 1)
        proportion = max(proportion, 0)

    left = speed*proportion
    right = speed - left

    if left > 100:
        warnings.append({'left': left})
        left = 100
        
    if right > 100:
        warnings.append({'right': right})
        right = 100
        
    return left, right, warnings

