from typing_extensions import deprecated
from components import Component
import cv2
import numpy as np
from typing import List, Dict, Literal, Union, Optional, Any
from data_collection.data_collection import LoggerSet, Logger
from multiprocessing import Process
from multiprocessing.sharedctypes import Synchronized as SharedValue


import sys
sys.path.append("/usr/lib/python3/dist-packages")
from picamera2 import Picamera2
from libcamera import Transform # type: ignore

class PicameraV2(Component):
    def __init__(self, resolution, framerate, config_overrides,  logger:Logger, ):
        """
        64 is the minimum it can go in each dimension

        114, 64 
        """
        picam2 = Picamera2()

        presets = dict( main={"size":resolution, "format": "RGB888"}, queue=False, controls={"FrameDurationLimits": (int(1e6/framerate), int(1e6/framerate))}, transform=Transform(hflip=True, vflip=True), buffer_count=1)
        for k, v in config_overrides.items():
            presets[k] = v


        video_config = picam2.create_video_configuration(**presets) # type: ignore
        picam2.configure(video_config) # type: ignore
        picam2.start()


        self.cap = picam2
        self.logger = logger

    def step(self) -> Union[None, np.ndarray]:
        img = self.cap.capture_array("main")[:, :, :3] # type: ignore 

        self.logger.log_time('time')
        self.logger.save_video_frame(img)
        return img

    @classmethod
    def create_shared_outputs(cls) -> Component.SHARED_VARIABLE_LIST:
        return [None]

    
    @classmethod
    def entry(
        cls, 
        resolution=0, framerate=0, config_overrides:Any={}, 
        logger_set: Optional[LoggerSet]=None, 
        *args, **kwargs

        ):

        assert isinstance(logger_set, LoggerSet)
        logger = logger_set.get_logger(**kwargs)

        return cls(resolution, framerate, config_overrides, logger)


@deprecated('use V2')
class Picamera:
    def __init__(self, resolution, framerate, config_overrides,  logger:Logger, ):
        """
        64 is the minimum it can go in each dimension

        114, 64 
        """
        picam2 = Picamera2()

        presets = dict( main={"size":resolution, "format": "RGB888"}, queue=False, controls={"FrameDurationLimits": (int(1e6/framerate), int(1e6/framerate))}, transform=Transform(hflip=True, vflip=True), buffer_count=1)
        for k, v in config_overrides.items():
            presets[k] = v


        video_config = picam2.create_video_configuration(**presets) # type: ignore
        picam2.configure(video_config) # type: ignore
        picam2.start()


        self.cap = picam2
        self.logger = logger

    def step(self) -> Union[None, np.ndarray]:
        img = self.cap.capture_array("main")[:, :, :3] # type: ignore 

        self.logger.log_time('time')
        self.logger.save_video_frame(img)
        return img


    @staticmethod
    def main(resolution, framerate, config_overrides, logger: Logger): 
        logger.setup_video_saver(resolution=resolution, framerate=framerate)

        component = Picamera(resolution, framerate, config_overrides, logger)
        
        while True:
            logger.increment_idx()
            component.step()
            
            
    @staticmethod
    def start(resolution, framerate, config_overrides, logger_set: LoggerSet, **kwargs):

        logger = logger_set.get_logger(**kwargs)
        process = Process(target=Picamera.main, args=(resolution, framerate, config_overrides, logger))
        process.start()
        return process


@deprecated('use V2 (not implemented yet)')
class Camera:
    def __init__(self, camera_idx, logger:Logger):
        self.cap = cv2.VideoCapture(camera_idx)
        self.logger = logger

    def step(self) -> Union[None, np.ndarray]:
        ret, img = self.cap.read()

        if ret:
            self.logger.log_time('time')
            self.logger.save_video_frame(img)
            return img
        else:
            return None

    @staticmethod
    def main(camera_idx, logger: Logger): 
        logger.setup_video_saver()

        component = Camera(camera_idx, logger)
        
        while True:
            logger.increment_idx()
            component.step()
            
            
    @staticmethod
    def start(camera_idx, logger_set: LoggerSet, **kwargs):

        logger = logger_set.get_logger(**kwargs)
        process = Process(target=Camera.main, args=(camera_idx, logger))
        process.start()
        return process