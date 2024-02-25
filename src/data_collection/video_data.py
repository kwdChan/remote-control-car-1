from pathlib import Path
from typing import List, Dict, Literal, Tuple, Union, cast
import datetime, time 
import numpy as np 
import pandas as pd
import pickle
import cv2

                    
class VideoSaver:

    DEFAULT_FOURCC = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    def __init__(self, path, save_interval=450, fourcc=DEFAULT_FOURCC, framerate=24, resolution=(640, 480)):

        path = Path(path)
        path.mkdir(exist_ok=True)

        self.path = path
        self.save_interval = save_interval
        self.resolution = resolution
        self.framerate = framerate
        self.fourcc = fourcc

        # states 
        self.video_idx = -1
        self.frame_idx = -1
        self.video_writer: cv2.VideoWriter

        # start 
        self.new_video()

    def new_video(self):

        self.video_idx += 1

        current_video_path = self.path / f"{self.video_idx}.mp4"
        self.video_writer = cv2.VideoWriter(str(current_video_path), self.fourcc, self.framerate, self.resolution  )
        self.frame_idx = -1
        



    def save_frame(self, arr:np.ndarray) -> Tuple[int, int, float]:
        timer = time.monotonic()
        self.frame_idx +=1 
        try: 
            self.video_writer.write(arr)
        except Exception as e:
            self.video_writer.release()
            raise e

        to_return = (self.video_idx, self.frame_idx, timer)

        if self.frame_idx >= self.save_interval: 
            self.video_writer.release()
            self.new_video()

        return to_return
    


        