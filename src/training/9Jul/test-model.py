raise NotImplementedError("the fixed verison is in the ipynb")
import time
from typing import Callable, Tuple, TypeVar, cast, get_origin, get_args, Union, Any, Optional, Dict, List
import sys
import os
sys.path.insert(0, '.') 

from components import CallChannel, ComponentStarter

from components.syncronisation import ProcessStarter
from components.two_wheels import TwoWheelsV3
from components.logger import LoggerComponent
from components.controller.bluetooth_controller import BlueToothCarControlSPP_V2
from components.camera import Picamera2V2

from components import (
    TwoWheelsV3, 
    LoggerComponent, 
    BlueToothCarControlSPP_V2, 
    start_bluetooth_server_v2b, 
    AngularSpeedControlV3, 
    Picamera2V2,
    FrameMLToAngularVelocity, 
    get_switches, get_switch
)

from data_collection.data_collection import LoggerSet
from multiprocessing import Manager
import numpy as np
import datetime
import RPi.GPIO as GPIO
import sys
import signal
import atexit

GPIO.setmode(GPIO.BOARD)

import tensorflow.keras as keras # type: ignore 

def get_model3(output_idx = 0):
    # angle difference [200, 400, 800]
    model_path = './training/9Jul/model3-10Jul.keras'
    ml_model = keras.models.load_model(model_path) 
    def model(arr):
        out = ml_model([arr[None, :]])[0, output_idx]
        return out*np.pi

    return model

def get_model4(output_idx = 2):
    # angular velocity [50, 100, 200, 400]
    model_path = './training/9Jul/model4-10Jul.keras'
    ml_model = keras.models.load_model(model_path) 

    def model(arr):

        out = ml_model([arr[None, :]])[0][output_idx]
        return out*100

    return model

def start_everything(): 


    loggerset = LoggerSet('./log/model_test'+str(datetime.datetime.now()), overwrite_ok=False)
    manager = Manager()

    # create 
    logger_process = ComponentStarter(
        LoggerComponent, 
        manager, 
        init_kwargs=dict(
            loggerset = loggerset
        ),
        loop_intervals={'step': 1/100},
    )

    two_wheel_process = ComponentStarter(
        TwoWheelsV3, 
        manager, 
        init_kwargs=dict(
            left_pin = 33, 
            right_pin = 32, 
            dir_pins = (16, 18), 
            name='TwoWheelsV3'
        ),
        loop_intervals={'step': 1/100},
        instantiator = TwoWheelsV3.entry
    )

    angular_speed_control_process = ComponentStarter(
        AngularSpeedControlV3, 
        manager, 
        init_kwargs=dict(
            i2c_address=0x68, 
            bus_num=1,
            name='AngularSpeedControlV3'
        ),
        loop_intervals={'step': 1/100},
        instantiator = AngularSpeedControlV3.entry
    )

    bluetooth_control_process = ComponentStarter(
        BlueToothCarControlSPP_V2, 
        manager, 
        init_kwargs={},
        loop_intervals={'step': 1/100},
    )

    bt_ser_out, bt_ser_process_man = start_bluetooth_server_v2b(manager)

    picamera_process = ComponentStarter(
        Picamera2V2, 
        manager, 
        init_kwargs=dict(
            resolution=(114, 64), 
            framerate=30,
            name='Picamera2V2'
        ),
        loop_intervals={'step': 1/30},
        sample_setup_kwargs=dict(default_values=[np.zeros((64, 114, 3), dtype=np.uint8)])
    )

    ml_process = ComponentStarter(
        FrameMLToAngularVelocity,
        manager, 
        init_kwargs=dict(
            func=get_model4, 
            name='FrameMLToAngularVelocity', 
            delay_ms=200
        ),
    )

    ## RPCs
    two_wheel_process.register_outgoing_rpc(
        dict(log=logger_process.incoming_rpcs['log'])
    )

    angular_speed_control_process.register_outgoing_rpc(
        dict(log=logger_process.incoming_rpcs['log'])
    )


    bluetooth_control_process.register_outgoing_rpc(
        dict(log=logger_process.incoming_rpcs['log'])
    )

    picamera_process.register_outgoing_rpc(
        dict(
            log=logger_process.incoming_rpcs['log'],
            setup_video_saver=logger_process.incoming_rpcs['setup_video_saver'],
            save_video_frame=logger_process.incoming_rpcs['save_video_frame'],
            notify_ml = ml_process.incoming_rpcs['run_model'],
            )
    )
    ml_process.register_outgoing_rpc(
        dict(log=logger_process.incoming_rpcs['log'])
    )





    ## Samples

    is_ml = lambda: bool(bt_ser_out[0]().get('start'))


    omega = get_switch(
        ml_process.outgoing_sample_readers[0], 
        angular_speed_control_process.outgoing_samples[1], 
        is_ml
    )
    speed = get_switch(
        lambda: 100, 
        angular_speed_control_process.outgoing_samples[0], 
        is_ml
    )


    two_wheel_process.register_incoming_samples(
        angular_speed_control_process.outgoing_samples
    )

    angular_speed_control_process.register_incoming_samples(
        [speed, omega]
    )

    ml_process.register_incoming_samples(
        picamera_process.outgoing_samples
    )
    
    bluetooth_control_process.register_incoming_samples(
        bt_ser_out
    )

    # Start
    bt_ser_process_man.start()
    bluetooth_control_process.start()
    logger_process.start()
    two_wheel_process.start()
    angular_speed_control_process.start()
    picamera_process.start()
    ml_process.start()


    # return
    result = (
        bt_ser_process_man, 
        bluetooth_control_process.process_starter, 
        logger_process.process_starter, 
        two_wheel_process.process_starter, 
        angular_speed_control_process.process_starter, 
        picamera_process.process_starter, 
        ml_process.process_starter
    )

    result = cast(List[ProcessStarter], result)
    def signal_handler(signum, frame):

        for r in result:
            r.kill()
    atexit.register(signal_handler, None, None)

    # there can only be one signal_handler. doing so overwrite the handler outside 
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)


    return result

def retry_everything(n_times, mon_interval_sec):


    def signal_handler(signum, frame):
        sys.exit()
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)    

    for n in range(n_times): 
        process_starters = start_everything() 
        while all((p.is_alive() for p in process_starters)): 
            time.sleep(mon_interval_sec)

        [p.kill() for p in process_starters]

retry_everything(1, mon_interval_sec = 2)