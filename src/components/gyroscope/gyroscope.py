from typing import Any, Union, cast, overload
from typing_extensions import deprecated, override
from . import mpu6050
from data_collection.data_collection import Logger, LoggerSet
import time
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from ..utils import receive_latest, send, ReceiveLatest
from typing import Tuple, List

import numpy as np
from ahrs.filters import Madgwick
from ahrs import Quaternion

from components import Component

def from_axang(axis: np.ndarray, angle) -> np.ndarray:

    def normalise(arr):
        return arr / np.sqrt((arr**2).sum()) # type: ignore

    axis = normalise(axis)
    sine = np.sin(angle/2) # type: ignore
    cosine = np.cos(angle/2) # type: ignore

    return np.r_[cosine, sine*axis]

class OrientationTrackerV2(Component):
    def __init__(self, device: mpu6050.MPU6050, logger: Logger):
        self.device = device
        self.madgwick = Madgwick()
        self.logger = logger

        self.initial_ori, self.gyro_bias = self.get_initial_orientation_and_gyro_bias(device)


        logger.log("initial_ori", self.initial_ori)
        logger.log("gyro_bias", self.gyro_bias)

        # states
        self.t_last = time.monotonic()
        self.last_ori = self.initial_ori
    
    @override
    def step(self) -> Tuple[float, float, float, float]:
        new_time = time.monotonic()
        self.madgwick.Dt =  new_time - self.t_last
        self.t_last = new_time

        acc = self.device.get_accel_data(g=True, return_dict=False)
        gyr = (self.device.get_gyro_data(return_dict=False)-self.gyro_bias)/180*np.pi
        
        new_ori = self.madgwick.updateIMU(self.last_ori, gyr=gyr, acc=acc)
        self.logger.log_time("time_OrientationTracker")
        self.logger.log('q', new_ori)
        
        self.last_ori = new_ori


        return new_ori[0], new_ori[1], new_ori[2], new_ori[3]  #, gyr, acc


        new_ori = Quaternion(new_ori)
        
        return new_ori#, gyr, acc
    
    @classmethod
    def get_initial_orientation_and_gyro_bias(cls, device, sec=1) -> Tuple[np.ndarray, np.ndarray]:

        def calculate(static_acc):
            ref = np.array([0, 0, 1])

            angle = np.arccos(np.dot(static_acc, ref)) # type:ignore 
            axis = np.cross(static_acc, ref)

            return from_axang(axis, angle)

        _t = time.monotonic()
        accs = []
        gyros = []
        while (time.monotonic() - _t) < sec:
            accs.append(device.get_accel_data(g=True, return_dict=False))
            gyros.append(device.get_gyro_data(return_dict=False))

        acc_mean = np.array(accs).mean(0)
        initial_q = calculate(acc_mean)
        gyro_bias = np.array(gyros).mean(0)
        return initial_q, gyro_bias

class OrientationTracker:
    def __init__(self, device: mpu6050.MPU6050, logger: Logger):
        self.device = device
        self.madgwick = Madgwick()
        self.logger = logger

        self.initial_ori, self.gyro_bias = OrientationTracker.get_initial_orientation_and_gyro_bias(device)


        logger.log("initial_ori", self.initial_ori)
        logger.log("gyro_bias", self.gyro_bias)

        # states
        self.t_last = time.monotonic()
        self.last_ori = self.initial_ori

    def step(self):
        new_time = time.monotonic()
        self.madgwick.Dt =  new_time - self.t_last
        self.t_last = new_time

        acc = self.device.get_accel_data(g=True, return_dict=False)
        gyr = (self.device.get_gyro_data(return_dict=False)-self.gyro_bias)/180*np.pi
        
        new_ori = self.madgwick.updateIMU(self.last_ori, gyr=gyr, acc=acc)
        self.logger.log_time("time_OrientationTracker")
        self.logger.log('q', new_ori)
        
        self.last_ori = new_ori
        new_ori = Quaternion(new_ori)
        
        return new_ori#, gyr, acc
    
    @staticmethod
    def get_initial_orientation_and_gyro_bias(device, sec=1) -> Tuple[np.ndarray, np.ndarray]:

        def calculate(static_acc):
            ref = np.array([0, 0, 1])

            angle = np.arccos(np.dot(static_acc, ref)) # type:ignore 
            axis = np.cross(static_acc, ref)

            return from_axang(axis, angle)

        _t = time.monotonic()
        accs = []
        gyros = []
        while (time.monotonic() - _t) < sec:
            accs.append(device.get_accel_data(g=True, return_dict=False))
            gyros.append(device.get_gyro_data(return_dict=False))

        acc_mean = np.array(accs).mean(0)
        initial_q = calculate(acc_mean)
        gyro_bias = np.array(gyros).mean(0)
        return initial_q, gyro_bias


class AngularSpeedControl:
    def __init__(self, ori_tracker: OrientationTracker, logger: Logger):
        self.ori_tracker = ori_tracker
        self.ori0 = Quaternion(ori_tracker.initial_ori)
        self.logger = logger

        # states
        self.target_angle = 0 
        self.last_t = time.monotonic()
        self.last_angle = 0
        self.current_proportion = 0.5
    
    def step(self, degree_per_second, speed):
        ori = self.ori_tracker.step()

        this_t = time.monotonic() 
        time_passed = this_t - self.last_t
        target_angle = self.target_angle + degree_per_second*time_passed

        axis, angle = Quaternion(ori * self.ori0.conj).to_axang()
        if axis[-1]<0:
            angle *= -1
        angle = cast (float, np.rad2deg(angle)) # type:ignore


        angle_diff = (angle-target_angle)
        angle_diff = (angle_diff + 180) % 360 - 180

        # different to the given degree_per_second
        angular_velocity = (angle - self.last_angle)
        angular_velocity = (angular_velocity + 180) % 360 - 180
        angular_velocity = angular_velocity/time_passed

        
        k1=5/360
        k2=1/360
        #new_proportion = self.current_proportion + (k1*angle_diff + k2*angular_velocity)
        new_proportion = 0.5 + (k1*angle_diff + k2*(angular_velocity-degree_per_second))

        left, right, warn = speed_proportion_control(new_proportion, speed)


        self.last_t = this_t
        self.target_angle = target_angle
        self.last_angle = angle 
        self.current_proportion = new_proportion

        self.logger.log_time("time_AngularSpeedControl")
        self.logger.log("angle_diff", angle_diff)
        self.logger.log("axis", axis)
        self.logger.log("angle", angle)
        self.logger.log("target_angle", target_angle)
        self.logger.log("degree_per_second", degree_per_second)
        self.logger.log("speed", speed)
        self.logger.log("new_proportion", new_proportion)
        self.logger.log("left", left)
        self.logger.log("right", right)
        self.logger.log("warn", warn)
        self.logger.log("angular_velocity", angular_velocity)

        return left, right

    @staticmethod
    def main(sender_conn: Connection, receiver_conn: Connection, logger_set:LoggerSet, logger_name, i2c_address=0x68, bus_num=1):


        device = mpu6050.MPU6050(i2c_address, bus_num)
        logger = logger_set.get_logger(logger_name)


        tracker = OrientationTracker(device, logger)
        control = AngularSpeedControl(tracker, logger)

        receiver = ReceiveLatest(receiver_conn, logger, (0, 0))
        while True:
            logger.increment_idx()
            dps, speed = receiver.get()
            left, right = control.step(dps, speed)
            send((left, right), sender_conn, logger)


    @staticmethod
    def start(logger_set:LoggerSet, logger_name, i2c_address=0x68, bus_num=1):

        in_receiver, in_sender = Pipe(False)     
        out_receiver, out_sender = Pipe(False)     

        process = Process(target=AngularSpeedControl.main, args=(
            out_sender, in_receiver, logger_set, logger_name, i2c_address, bus_num)
            )
        process.start()
        return process, in_sender, out_receiver

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

@deprecated("bad result")
class GyroscopeWheelInput:
    def __init__(self, interval, address, bus, logger: Logger):
        self.gyroscope = Gyroscope(address, bus, logger)
        self.interval = interval

        # states
        self.angle = 0
        self.last_sample_monotonic: float = -1

    def step(self, angle, factor, speed):
        "angle is the target of the spin"
        assert angle == 0


        data = self.gyroscope.step()  
        spin = data['x'] # type: ignore

        #too Left: x positive 
        #too right: x negative 

        p = (spin - angle) * factor + 0.5

        left, right, warning = speed_proportion_control(p, speed)


        return left, right
    
    @staticmethod
    def main(interval, address, bus, logger, receiver: Connection, sender: Connection):
        component = GyroscopeWheelInput(interval, address, bus, logger)

        last_data = 0, 0, 0
        while True:
            logger.increment_idx()
            time.sleep(interval)

            data = receive_latest(receiver, logger, last_data) 
            if not data is None: 
                angle, factor, speed = data
                l, r = component.step(angle, factor, speed)
                send((l,r), sender, logger)
                last_data = data
                

    @staticmethod
    def start(interval, address, bus, logger_set: LoggerSet, **kwargs):

        logger = logger_set.get_logger(**kwargs) # type: ignore 
        in_receiver, in_sender = Pipe(False)     
        out_receiver, out_sender = Pipe(False)     

        process = Process(target=GyroscopeWheelInput.main, args=(interval, address, bus, logger, in_receiver, out_sender))
        process.start()
        return process, in_sender, out_receiver

    
@deprecated("bad result")
class Gyroscope:
    def __init__(self, address, bus, logger: Logger):
        self.device = mpu6050.MPU6050(address, bus)
        self.logger = logger 

    def step(self):
        now = time.monotonic()

        data = self.device.get_gyro_data()
        data['monotonic'] = now # type: ignore
        
        self.logger.log_time('gyroscope')
        self.logger.log('x', data['x']) # type: ignore
        self.logger.log('y', data['y']) # type: ignore
        self.logger.log('z', data['z']) # type: ignore

        return data

