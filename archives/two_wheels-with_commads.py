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


@deprecated("won't be tested for updates")
class TwoWheelsOnCommandV3:
    def __init__(self, ch_left, ch_right, logger:Logger):
        
        self.two_wheels = TwoWheels(ch_left, ch_right, logger)
        self.logger = logger
        
        
        # state 
        self.speed = 25
        self.proportion = 0.5
        self.command_status = {}


    def step(self, command_update:Dict[str, bool]) -> Tuple[float, float]:

        for k, v in command_update.items():
            self.command_status[k] = v
            

        go = False
        for command, active in self.command_status.items():
            if not active: continue 

            if command == 'L+':
                self.proportion -= 0.05
                self.command_status[command] = False

            
            elif command == 'R+':
                self.proportion += 0.05       
                self.command_status[command] = False 
                
            elif command == 'LR':
                self.proportion = 0.5    

            elif command == 'V+':
                self.speed += 5   
                self.command_status[command] = False 

            elif command == 'V-':
                self.speed -= 5   
                self.command_status[command] = False
            
            elif command == 'go':
                go = True

            else: 
                pass

        if go:
            left, right, warnings = speed_proportion_control(self.proportion, self.speed)
            self.two_wheels.step(left, right)
            self.logger.log('going', True)
            self.logger.log('warnings', warnings)
        else:
            self.logger.log('going', False)
            self.two_wheels.step(0, 0)
            
        self.logger.log('command_update', command_update)
        self.logger.log('command', self.command_status)
        self.logger.log_time('time')

        return self.proportion, self.speed
    
    @staticmethod
    def sample(input_con: Connection) -> Dict[str, bool]:
        """
        read only the latest input

        the component stale if the inputs are comming faster than it can read
        """
        command_update = {}
        while input_con.poll():
            command_update = input_con.recv()
        return command_update
         
    @staticmethod
    def main(left_pin:int, right_pin:int, logger: Logger, input_con: Connection):
        
        GPIO.setmode(GPIO.BOARD)
        ch_left = setup_pwm(left_pin,freq=300)
        ch_right = setup_pwm(right_pin,freq=300)

        component = TwoWheelsOnCommandV3(ch_left, ch_right, logger)

        while True:
            logger.increment_idx()
            command = TwoWheelsOnCommandV3.sample(input_con)
            component.step(command)

            time.sleep(0.1)

    @staticmethod
    def local(left_pin, right_pin, logger_set: LoggerSet, **kwargs):

        GPIO.setmode(GPIO.BOARD)
        ch_left = setup_pwm(left_pin,freq=300)
        ch_right = setup_pwm(right_pin,freq=300)
        
        logger = logger_set.get_logger(**kwargs)

        component = TwoWheelsOnCommandV3(ch_left, ch_right, logger)

        return component

    @staticmethod
    def start(left_pin, right_pin, logger_set: LoggerSet, **kwargs):
        """
        using GPIO.BOARD pin
        """

        logger = logger_set.get_logger(**kwargs)
        receiver, sender = Pipe(False)        
        process = Process(target=TwoWheelsOnCommandV3.main, args=(left_pin, right_pin, logger, receiver))
        process.start()
        return process, sender


@deprecated("won't be tested for updates")
class TwoWheelsOnCommandV2:
    def __init__(self, ch_left, ch_right, logger:Logger, n_sec_per_press=0.5):
        
        self.two_wheels = TwoWheels(ch_left, ch_right, logger)
        self.logger = logger
        self.n_sec_per_press = n_sec_per_press
        
        
        # state 
        self.last_pressed = -1 # time when the 
        self.speed = 25
        self.proportion = 0.5


    def step(self, command:str) -> Tuple[float, float]:
        
        if command == 'L+':
            self.proportion -= 0.05
        
        elif command == 'R+':
            self.proportion += 0.05        
            
        elif command == 'LR':
            self.proportion = 0.5    
        
        elif command == 'V+':
            self.speed += 5    

        elif command == 'V-':
            self.speed -= 5   
        
        elif command == 'go':
            self.last_pressed = time.monotonic()          

        elif command == 'stop':
            self.last_press = -1
        
        elif command == '.':
            pass
        
        else: 
            pass

        if  (time.monotonic()  - self.last_pressed) < self.n_sec_per_press:
            left, right, warnings = speed_proportion_control(self.proportion, self.speed)
            self.two_wheels.step(left, right)

            self.logger.log('going', True)
            self.logger.log('warnings', warnings)
        else:
            self.logger.log('going', False)
            self.two_wheels.step(0, 0)
            

        self.logger.log('command', command)
        self.logger.log_time('time')

        return self.proportion, self.speed
    
    @staticmethod
    def sample(input_con: Connection) -> str:
        """
        read only the latest input

        the component stale if the inputs are comming faster than it can read
        """
        command = ''
        while input_con.poll():
            command = input_con.recv()
        return command
         
    @staticmethod
    def main(left_pin:int, right_pin:int, logger: Logger, input_con: Connection):
        
        GPIO.setmode(GPIO.BOARD)
        ch_left = setup_pwm(left_pin,freq=300)
        ch_right = setup_pwm(right_pin,freq=300)

        component = TwoWheelsOnCommandV2(ch_left, ch_right, logger)

        while True:
            logger.increment_idx()
            command = TwoWheelsOnCommandV2.sample(input_con)
            component.step(command)

            time.sleep(0.1)

    @staticmethod
    def local(left_pin, right_pin, logger_set: LoggerSet, **kwargs):

        GPIO.setmode(GPIO.BOARD)
        ch_left = setup_pwm(left_pin,freq=300)
        ch_right = setup_pwm(right_pin,freq=300)

        logger = logger_set.get_logger(**kwargs)

        component = TwoWheelsOnCommandV2(ch_left, ch_right, logger)

        return component

    @staticmethod
    def start(left_pin, right_pin, logger_set: LoggerSet, **kwargs):
        """
        using GPIO.BOARD pin
        """

        logger = logger_set.get_logger(**kwargs)
        receiver, sender = Pipe(False)        
        process = Process(target=TwoWheelsOnCommandV2.main, args=(left_pin, right_pin, logger, receiver))
        process.start()
        return process, sender

    @staticmethod
    def ircode2command(code, unknown='.'):
        lookup = {
            16750695: 'go',   # 100+
            16769055: 'V-',   # -
            16754775: 'V+',   # +
            16720605: 'L+',   # |<<  R Wheel speed up = steer left
            16712445: 'R+',   # >>|
            16761405: 'LR',   # >||
            16748655: 'stop', # EQ
        }
        if code in lookup:
            return lookup[code]
        else: 
            return unknown



@deprecated("won't be tested for updates")
class TwoWheelsOnCommand:
    def __init__(self, ch_left, ch_right, logger:Logger):
        
        self.two_wheels = TwoWheels(ch_left, ch_right, logger)
        self.logger = logger
        
        # state 
        self.speed = 0
        self.proportion = 0.5

    


    def step(self, command:str) -> None:
        
        if command == 'stop':
            self.speed = 0
            
        elif command == 'L+':
            self.proportion -= 0.05
        
        elif command == 'R+':
            self.proportion += 0.05        
        
        elif command == 'V+':
            self.speed += 10    
            
        elif command == 'V-':
            self.speed -= 5    
            
        elif command == 'LR':
            self.proportion = 0.5    

        elif command == '.':
            pass
            
        else: 
            pass

        left, right, warnings = speed_proportion_control(self.proportion, self.speed)
        self.two_wheels.step(left, right)
        
        self.logger.log('warnings', warnings)
        self.logger.log('command', command)
        self.logger.log_time('time')
    
    @staticmethod
    def sample(input_con: Connection) -> str:
        """
        read only the latest input

        the component stale if the inputs are comming faster than it can read
        """
        command = ''
        while input_con.poll():
            command = input_con.recv()
        return command
         
    @staticmethod
    def main(left_pin:int, right_pin:int, logger: Logger, input_con: Connection):
        
        GPIO.setmode(GPIO.BOARD)
        ch_left = setup_pwm(left_pin,freq=300)
        ch_right = setup_pwm(right_pin,freq=300)

        component = TwoWheelsOnCommand(ch_left, ch_right, logger)

        while True:
            logger.increment_idx()
            command = TwoWheelsOnCommand.sample(input_con)
            component.step(command)

            time.sleep(0.1)

    @staticmethod
    def start(left_pin, right_pin, logger_set: LoggerSet, **kwargs):
        """
        using GPIO.BOARD pin
        """

        logger = logger_set.get_logger(**kwargs)
        receiver, sender = Pipe(False)        
        process = Process(target=TwoWheelsOnCommand.main, args=(left_pin, right_pin, logger, receiver))
        process.start()
        return process, sender

    @staticmethod
    def ircode2command(code, unknown='.'):
        lookup = {
            16769055: 'V-',   
            16754775: 'V+',
            16720605: 'L+',   # R Wheel speed up = steer left
            16712445: 'R+',  
            16761405: 'LR',
            16748655: 'stop',
        }
        if code in lookup:
            return lookup[code]
        else: 
            return unknown


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

