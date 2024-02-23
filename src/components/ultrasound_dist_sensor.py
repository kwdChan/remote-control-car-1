#from gpiozero import LED
import RPi.GPIO as GPIO
import time
from datetime import datetime, timedelta
from data_collection.data_collection import LoggerSet, Logger

from multiprocessing.connection import Connection
from multiprocessing import Pipe, Process

class DistanceSensorUrgentStop:
    def __init__(self, pin_trigger:int, pin_echo:int, threshold: float, logger: Logger):

        self.ultrasonic_distance_sensor = UltrasonicDistanceSensor(pin_trigger, pin_echo, logger)
        self.threshold = threshold

    def step(self) -> bool:
        distance = self.ultrasonic_distance_sensor.step()
        return distance < self.threshold

    @staticmethod
    def main(pin_trigger:int, pin_echo:int, threshold: float, logger_set: LoggerSet, sender: Connection, **kwargs):

        logger = logger_set.get_logger(**kwargs)
        device = DistanceSensorUrgentStop(pin_trigger, pin_echo, threshold, logger)

        while True:
            sender.send(device.step())
            time.sleep(0.1)

    @staticmethod
    def start(pin_trigger:int, pin_echo:int, threshold: float, logger_set: LoggerSet, **kwargs):
        receiver, sender = Pipe(False)
        p = Process(target=DistanceSensorUrgentStop.main, args=(pin_trigger, pin_echo, threshold, logger_set, sender), kwargs=kwargs)
        p.start()
        return p, receiver

        



class UltrasonicDistanceSensor:
    def __init__(self, pin_trigger:int, pin_echo:int, logger: Logger):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin_echo, GPIO.IN,  pull_up_down=GPIO.PUD_UP)
        GPIO.setup(pin_trigger, GPIO.OUT)

        self.pin_trigger = pin_trigger
        self.pin_echo = pin_echo
        self.logger = logger

    def step(self) -> float:
        trig_pin = self.pin_trigger
        echo_pin = self.pin_echo
        
        GPIO.output(trig_pin, GPIO.LOW)
        GPIO.output(trig_pin, GPIO.HIGH)
        time.sleep(20e-6) # the device is outputting 
        GPIO.output(trig_pin, GPIO.LOW)       
        
        while not GPIO.input(echo_pin):
            pass

        start_time = time.monotonic()

        while GPIO.input(echo_pin):
            pass

        time_diff = (time.monotonic() - start_time)
        distance_m = time_diff*340/2

        self.logger.log('ultrasonic_distance_m', distance_m)
        return distance_m



