

from typing import List, Optional
from components.controller.bluetooth_controller import BlueToothCarControlSPP
from components.controller.bluetooth_SPP_server import start_bluetooth_server_v2
from components.gyroscope.gyroscope import AngularSpeedControlV2
from components.two_wheels import TwoWheelsV2
from components.camera import PicameraV2
from data_collection.data_collection import LoggerSet
from multiprocessing import Manager, Process
import time
from components import default_loop_v2, default_component_process_starter_v2
import atexit

import RPi.GPIO as GPIO 
GPIO.setmode(GPIO.BOARD) # type: ignore
import sys
log_path = sys.argv[1]
if log_path: 
    logger_set = LoggerSet(log_path, overwrite_ok=False)
else: 
    logger_set = LoggerSet('../log/temp_test', overwrite_ok=True)
manager = Manager()

w_out, w_starter = default_component_process_starter_v2(
    TwoWheelsV2, 
    init_kwargs= dict(
        left_pin=33, 
        right_pin=32, 
        dir_pins=(16, 18), 
        logger_set=logger_set, 
        name='TwoWheelsV2'
    ), 
    mainloop = default_loop_v2, 
    main_kwargs=dict(interval=1/50),
    manager=manager, 
)
imu_out, imu_starter = default_component_process_starter_v2(
    AngularSpeedControlV2, 
    init_kwargs= dict(
        logger_set=logger_set, 
        name='AngularSpeedControlV2'
    ), 
    mainloop = default_loop_v2, 
    main_kwargs=dict(interval=1/50),
    manager=manager, 
)
cam_out, cam_starter = default_component_process_starter_v2(
    PicameraV2, 
    init_kwargs= dict(
        resolution=(114, 64), framerate=30,
        logger_set=logger_set, 
        name='PicameraV2'
    ), 
    mainloop = default_loop_v2, 
    main_kwargs=dict(interval=1/50),
    manager=manager, 
    shared_outputs_kwargs=dict(resolution=(114, 64))
)

bt_ser_out, bt_ser = start_bluetooth_server_v2(manager)

bt_out, bt_starter = default_component_process_starter_v2(
    BlueToothCarControlSPP, 
    init_kwargs= dict(
        logger_set=logger_set, 
        name='BlueToothCarControlSPP'
    ), 
    mainloop=default_loop_v2, 
    main_kwargs=dict(interval=1/30), 
    manager=manager, 
)


w = w_starter(imu_out)
imu = imu_starter(bt_out)
cam = cam_starter()
bt = bt_starter(bt_ser_out)

ALL_PROCESSES = [bt_ser, w, imu, cam, bt]

def health_check(processes: List[Process], interval=1, fail_cbs=[], ok_cbs=[]):


    while True: 
        time.sleep(interval)
        for p in processes:
            if p.is_alive():
                for cb in ok_cbs:
                    cb(p)
            else: 
                for cb in fail_cbs:
                    cb(p)

                interval=1

def on_termination(p: Optional[Process]):
    print(p)
    print('failed')
    for prc in ALL_PROCESSES:
        prc.terminate()
    sys.exit(1)
atexit.register(on_termination, None)

health_check(ALL_PROCESSES, fail_cbs=[on_termination])
