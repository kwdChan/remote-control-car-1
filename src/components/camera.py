import cv2
import numpy as np
from typing import List, Dict, Literal, Union
from data_collection.data_collection import LoggerSet, Logger
from multiprocessing import Process

class Camera:
    def __init__(self, camera_idx, logger:Logger):
        self.cap = cv2.VideoCapture(camera_idx)
        self.logger = logger

    def step(self) -> Union[None, np.ndarray]:
        ret, img = self.cap.read()

        if ret:
            self.logger.log_time('time')
            self.logger.log('img', img)
            return img
        else:
            return None
    @staticmethod
    def main(camera_idx, logger: Logger): 

        component = Camera(camera_idx, logger)
        
        while True:
            logger.increment_idx()
            img = component.step()
            
    @staticmethod
    def start(camera_idx, logger_set: LoggerSet, **kwargs):

        logger = logger_set.get_logger(**kwargs)
        process = Process(target=Camera.main, args=(camera_idx, logger))
        process.start()
        return process