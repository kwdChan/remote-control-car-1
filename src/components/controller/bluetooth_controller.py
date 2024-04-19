
from typing import TypeVar, Union
import bluetooth
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from data_collection.data_collection import Logger, LoggerSet
import numpy as np 
from components.utils import receive_latest, send, receive_immediately
import time

class ServerForAppArduinoBlueControl:
    def __init__(self, logger: Logger):

        server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        server_sock.bind(("", bluetooth.PORT_ANY))
        print("Waiting for connections")
        server_sock.listen(1)  # type: ignore 

        client_sock, client_info = server_sock.accept()
        print("Accepted connection from", client_info)

        self.client_sock = client_sock
        self.server_sock = server_sock
        self.logger = logger
        
    def recv(self):
        data = self.client_sock.recv(1024) # type: ignore 
        return data.decode()

    @staticmethod
    def main(loggerset: LoggerSet, sender: Connection, **kwargs):

        logger = loggerset.get_logger(**kwargs)
        s = ServerForAppArduinoBlueControl(logger)

        commands = set()

        last_sent = time.monotonic()
        while True:
            commands.add(s.recv())
            if (time.monotonic() - last_sent) > 1/150:
                s.logger.increment_idx()
                send(commands, sender, s.logger)
                commands = set()

    @staticmethod
    def start(loggerset: LoggerSet, **kwargs):
        receiver, sender = Pipe(False)        
        process = Process(
            target=ServerForAppArduinoBlueControl.main, 
            args=(loggerset, sender), 
            kwargs = kwargs
        )
        process.start()
        return process, receiver

    def __del__(self):
        self.server_sock.close() # type: ignore 

class BlueToothCarControl:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.max_interval = 50e-3

        # states
        self.speed = 0
        self.angular_velocity = 0

        self.command_last = {}
        self.command_since = {}


    def step(self):
        current_time = time.monotonic()
        self.check_reset_command(current_time)

        if 'up' in self.command_since:
            up = current_time - self.command_since['up'] 
            self.speed = sigmoid(3*up) * 100 + 20
        else: 
            self.speed = max(0, self.speed-1)

        right = 0
        if 'right' in self.command_since:
            right = current_time - self.command_since['right'] 

        left = 0
        if 'left' in self.command_since:
            left = current_time - self.command_since['left'] 

        self.angular_velocity = (2*sigmoid(left - right)-1) * 70

        return  self.angular_velocity, self.speed

    def check_reset_command(self, current_time=None):
        if current_time is None: 
            current_time = time.monotonic()
        
        for command, last_time in self.command_last.items():
            if (current_time - last_time) > self.max_interval:

                if command in self.command_since: 
                    del self.command_since[command]
        
        
    def receive(self, commands: set): 

        current_time = time.monotonic()

        for command in commands: 
            self.command_last[command] = current_time

            if not command in self.command_since: 
                self.command_since[command] = current_time


    @staticmethod
    def main(
        loggerset: LoggerSet, 
        receiver: Connection, 
        sender: Connection, 
        **kwargs
    ):

        logger = loggerset.get_logger(**kwargs)
        c = BlueToothCarControl(logger)

        time_last_step = time.monotonic()
        while True: 
            command = receive_latest(receiver, logger, None)
            if not command is None: 
                c.receive(command)
            current_time = time.monotonic()
            if (current_time - time_last_step) > 1/150:
                data = c.step()
                send(data, sender, logger)
                logger.increment_idx()
                time_last_step = current_time

    @staticmethod
    def start(loggerset: LoggerSet, **kwargs):
        in_receiver, in_sender = Pipe(False)        
        out_receiver, out_sender = Pipe(False)        
        process = Process(
            target=BlueToothCarControl.main, 
            args=(loggerset, in_receiver, out_sender), 
            kwargs = kwargs
        )
        process.start()
        return process, in_sender, out_receiver

T = TypeVar('T')
def sigmoid(x: T) -> T:
    return 1/(1+np.exp(-x)) #type: ignore 


def get_value(time_list, current_time):
    t = current_time - np.array(time_list)
    half_life = 0.5
    return np.sum(1*2**(-t/half_life)) # type: ignore
    
