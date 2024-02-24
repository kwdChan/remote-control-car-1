from typing import List, Dict, Literal, Union, Tuple, Any
import datetime, time
from typing_extensions import deprecated 
import numpy as np 
import pandas as pd
import RPi.GPIO as GPIO
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from data_collection.data_collection import LoggerSet, Logger

def setup_pwm(pin, freq=300) -> GPIO.PWM:
    GPIO.setup(pin, GPIO.OUT)
    return GPIO.PWM(pin, freq) 

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
    def main(left_pin:int, right_pin:int, logger: Logger, input_con: Connection):
        
        GPIO.setmode(GPIO.BOARD)
        ch_left = setup_pwm(left_pin,freq=300)
        ch_right = setup_pwm(right_pin,freq=300)

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
    def start(left_pin, right_pin, logger_set: LoggerSet, **kwargs):
        """
        using GPIO.BOARD pin
        """

        logger = logger_set.get_logger(**kwargs)
        receiver, sender = Pipe(False)        
        process = Process(target=TwoWheels.main, args=(left_pin, right_pin, logger, receiver))
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

