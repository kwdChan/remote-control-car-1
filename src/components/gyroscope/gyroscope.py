from multiprocessing.managers import SyncManager
from typing import Any, Union, cast, Optional
from typing_extensions import deprecated, override
from . import mpu6050
from data_collection.data_collection import Logger, LoggerSet
import time
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from multiprocessing.managers import BaseProxy, BaseManager

from typing import Tuple, List
from ctypes import c_double
import numpy as np
from ahrs.filters import Madgwick
from ahrs import Quaternion

from components import Component, shared_value

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

    @classmethod
    def create_shared_outputs(cls, manager: BaseManager) -> List[Optional[BaseProxy]]:
        
        assert isinstance(manager, SyncManager)
        w = manager.Value(c_double, 0)
        x = manager.Value(c_double, 0)
        y = manager.Value(c_double, 0)
        z = manager.Value(c_double, 0)
        return [w,x,y,z]

@deprecated("Move to OrientationTrackerV2")
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


class AngularSpeedControlV2(Component):
    def __init__(self, ori_tracker: OrientationTrackerV2, logger: Logger):
        self.ori_tracker = ori_tracker
        self.ori0 = Quaternion(ori_tracker.initial_ori)
        self.logger = logger

        # states
        self.target_angle = 0 
        self.last_t = time.monotonic()
        self.last_angle = 0
        self.current_proportion = 0.5

    @override
    def step(
        self, 
        degree_per_second=0, 
        speed=0
    ) -> Tuple[float, float]:
        ori = Quaternion(self.ori_tracker.step())

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
        
    @override
    @classmethod
    def entry(
        cls, 
        logger_set:Optional[LoggerSet]=None, 
        name='', 
        i2c_address=0x68, 
        bus_num=1,
        **kwargs
    ):
        assert name, "name cannot be left empty"
        assert logger_set, "logger_set cannot be left empty"
        
        device = mpu6050.MPU6050(i2c_address, bus_num)
        logger = logger_set.get_logger(name, **kwargs)

        tracker = OrientationTrackerV2(device, logger)
        control = cls(tracker, logger)
        return control

    @classmethod
    def create_shared_outputs(cls, manager: BaseManager) -> List[Optional[BaseProxy]]:
        
        assert isinstance(manager, SyncManager)
        left = manager.Value(c_double, 0)
        right = manager.Value(c_double, 0)

        return [left, right]

    @classmethod
    def create_shared_outputs_rw(cls, manager: BaseManager):
        
        assert isinstance(manager, SyncManager)
        leftr, leftw = shared_value(manager, 'd')
        rightr, rightw = shared_value(manager, 'd')

        return [leftr, rightr], [leftw, rightw]

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
