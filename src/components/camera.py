import numpy as np
from typing import List, Dict, Literal, Tuple, Union, Optional, Any, Callable, cast

from components.syncronisation import ComponentInterface, CallChannel, component, sampler, samples_producer, rpc, declare_method_handler, loop
from components.logger import LoggerComponent, add_time
from picamera2 import Picamera2 
from libcamera import Transform # type: ignore




@component
class Picamera2V2(ComponentInterface):
    def __init__(
        self, resolution, framerate, 
        log, setup_video_saver, save_video_frame, notify_ml:Optional[CallChannel]=None, 
        config_overrides ={},  name="PicameraV2"):
        """
        64 is the minimum it can go in each dimension

        114, 64 
        """

        # RPC type check declaration 
        setup_video_saver = declare_method_handler(setup_video_saver, LoggerComponent.setup_video_saver)
        log = declare_method_handler(log, LoggerComponent.log)
        save_video_frame = declare_method_handler(save_video_frame, LoggerComponent.save_video_frame)
        if notify_ml: 
            notify_ml = cast(CallChannel[[], None], notify_ml) 

        # start
        picam2 = Picamera2()
        setup_video_saver.call(name, resolution=resolution, framerate=framerate)
        

        presets = dict( main={"size":resolution, "format": "RGB888"}, queue=False, controls={"FrameDurationLimits": (int(1e6/framerate), int(1e6/framerate))}, transform=Transform(hflip=True, vflip=True), buffer_count=1)
        for k, v in config_overrides.items():
            presets[k] = v


        video_config = picam2.create_video_configuration(**presets) # type: ignore
        picam2.configure(video_config) # type: ignore
        picam2.start()


        self.cap = picam2
        self.name = name


        # state
        self.idx = 0

        # RPCs
        self.log = log
        self.notify_ml = notify_ml
        self.save_video_frame = save_video_frame

    @loop
    @samples_producer(typecodes=['B'])
    def step(self):
        #self.increment_index.call(self.name)() # need to 
        
        img = self.cap.capture_array("main")[:, :, :3] # type: ignore 

        self.log.call_no_return(self.name, add_time({}), self.idx)
        self.save_video_frame.call_no_return(self.name, img, self.idx)
        if self.notify_ml: 
            self.notify_ml.call_no_return()

        self.idx += 1
        return (img, )


