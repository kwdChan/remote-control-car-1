
from typing import Callable, Tuple, TypeVar, cast, get_origin, get_args, Union, Any, Optional, Dict, List
from components import EventBroadcaster, MessageChannel, create_component_starter, loop

from components.two_wheels import TwoWheelsV3
from components.logger import LoggerComponent
from components.controller.bluetooth_controller import BlueToothCarControlSPP_V2
from components.controller.bluetooth_SPP_server import start_bluetooth_server_v2
from components.gyroscope.gyroscope import AngularSpeedControlV3
from components.camera import Picamera2V2
from data_collection.data_collection import LoggerSet
from multiprocessing import Manager


loggerset = LoggerSet("../log/test")
manager = Manager()

logger_starter = create_component_starter(
    LoggerComponent, 
    manager, 
    loop,  # doesn't need a loop....
    init_kwargs=dict(
        loggerset = loggerset
    ),
    loop_kwargs={'ideal_interval': 1/100},
)

two_wheel_starter = create_component_starter(
    TwoWheelsV3, 
    manager, 
    loop, 
    init_kwargs=dict(
        left_pin = 33, 
        right_pin = 32, 
        dir_pins = (16, 18), 
        name='TwoWheelsV3'
    ),
    loop_kwargs={'ideal_interval': 1/100},
    instantiater = TwoWheelsV3.entry

)

angular_speed_control_starter = create_component_starter(
    AngularSpeedControlV3, 
    manager, 
    loop, 
    init_kwargs=dict(
        i2c_address=0x68, 
        bus_num=1,
        name='AngularSpeedControlV3'
    ),
    loop_kwargs={'ideal_interval': 1/100},
    instantiater = AngularSpeedControlV3.entry
)

bluetooth_control_starter = create_component_starter(
    BlueToothCarControlSPP_V2, 
    manager, 
    loop, 
    init_kwargs={},
    loop_kwargs={'ideal_interval': 1/100},
)


picamera_starter = create_component_starter(
    Picamera2V2, 
    manager, 
    loop, 
    init_kwargs=dict(
        resolution=(114, 64), 
        framerate=30,
        name='Picamera2V2'
    ),
    loop_kwargs={'ideal_interval': 1/30},
)

bt_ser_out, bt_ser = start_bluetooth_server_v2(manager)

logger_starter.register_incoming_events(
    dict(
        log_handler=[
            two_wheel_starter.outgoing_events['logging'], 
            angular_speed_control_starter.outgoing_events['logging'], 
            bluetooth_control_starter.outgoing_events['logging'], 
            picamera_starter.outgoing_events['logging'], 
        ]
    )
)



two_wheel_starter.register_incoming_samples(
    angular_speed_control_starter.outgoing_samples
)

angular_speed_control_starter.register_incoming_samples(
    bluetooth_control_starter.outgoing_samples
)

bluetooth_control_starter.register_incoming_samples(
    bt_ser_out
)

logger_starter.start()
two_wheel_starter.start()
angular_speed_control_starter.start()
bluetooth_control_starter.start()
picamera_starter.start()

