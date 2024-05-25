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

