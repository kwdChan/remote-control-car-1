
from typing import Callable, Tuple, TypeVar, cast, get_origin, get_args, Union, Any, Optional, Dict, List
from components import CallChannel, ComponentStarter

from components.two_wheels import TwoWheelsV3
from components.logger import LoggerComponent
from components.controller.bluetooth_controller import BlueToothCarControlSPP_V2
from components.controller.bluetooth_SPP_server import start_bluetooth_server_v2
from components.gyroscope.gyroscope import AngularSpeedControlV3
from components.camera import Picamera2V2
from data_collection.data_collection import LoggerSet
from multiprocessing import Manager
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)


loggerset = LoggerSet("../log/test")
manager = Manager()

logger_starter = ComponentStarter(
    LoggerComponent, 
    manager, 
    init_kwargs=dict(
        loggerset = loggerset
    ),
    loop_intervals={'step': 1/100},
)

two_wheel_starter = ComponentStarter(
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

angular_speed_control_starter = ComponentStarter(
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

bluetooth_control_starter = ComponentStarter(
    BlueToothCarControlSPP_V2, 
    manager, 
    init_kwargs={},
    loop_intervals={'step': 1/100},
)


picamera_starter = ComponentStarter(
    Picamera2V2, 
    manager, 
    init_kwargs=dict(
        resolution=(114, 64), 
        framerate=30,
        name='Picamera2V2'
    ),
    loop_intervals={'step': 1/30},
)


bt_ser_out, bt_ser = start_bluetooth_server_v2(manager)

# sample
two_wheel_starter.register_incoming_samples(
    angular_speed_control_starter.outgoing_sample_readers
)

angular_speed_control_starter.register_incoming_samples(
    bluetooth_control_starter.outgoing_sample_readers
)

bluetooth_control_starter.register_incoming_samples(
    bt_ser_out
)

# rpcs
logger_triplet = dict(
    log=logger_starter.incoming_rpcs['log'], 
    log_time=logger_starter.incoming_rpcs['log_time'], 
    increment_index=logger_starter.incoming_rpcs['increment_index'], 
)

picamera_starter.register_outgoing_rpc(
    logger_starter.incoming_rpcs
)


two_wheel_starter.register_outgoing_rpc(
    logger_triplet
)
angular_speed_control_starter.register_outgoing_rpc(
    logger_triplet
)
bluetooth_control_starter.register_outgoing_rpc(
    logger_triplet
)




logger_starter.start()
two_wheel_starter.start()
angular_speed_control_starter.start()
bluetooth_control_starter.start()
picamera_starter.start()

