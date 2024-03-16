from typing import Any, Union
from . import mpu6050
from data_collection.data_collection import Logger, LoggerSet
import time
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from ..utils import receive_latest, send

class GyroscopeWheelInput:
    def __init__(self, interval, address, bus, logger: Logger):
        self.gyroscope = Gyroscope(address, bus, logger)
        self.interval = interval

        # states
        self.angle = 0
        self.last_sample_monotonic: float = -1

    def step(self, angle):
        self.gyroscope.step()    
        return 0, 0 
    
    @staticmethod
    def main(interval, address, bus, logger, receiver: Connection, sender: Connection):
        component = GyroscopeWheelInput(interval, address, bus, logger)

        last_angle = 0
        while True:
            logger.increment_idx()
            time.sleep(interval)

            angle = receive_latest(receiver, logger, last_angle)
        
            l, r = component.step(angle)
            send((l,r), sender, logger)
            last_angle = angle
            

    @staticmethod
    def start(interval, address, bus, logger_set: LoggerSet, **kwargs):

        logger = logger_set.get_logger(**kwargs)
        in_receiver, in_sender = Pipe(False)     
        out_receiver, out_sender = Pipe(False)     

        process = Process(target=GyroscopeWheelInput.main, args=(interval, address, bus, logger, in_receiver, out_sender))
        process.start()
        return process, in_sender, out_receiver




        

class Gyroscope:
    def __init__(self, address, bus, logger: Logger):
        self.device = mpu6050.MPU6050(address, bus)
        self.logger = logger 

    def step(self):
        now = time.monotonic()

        data = self.device.get_gyro_data()
        data['monotonic'] = now
        
        self.logger.log_time('gyroscope')
        self.logger.log('x', data['x'])
        self.logger.log('y', data['y'])
        self.logger.log('z', data['z'])

        return data

