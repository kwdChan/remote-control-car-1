from multiprocessing.managers import BaseManager
from typing_extensions import deprecated
from components import Component, shared_value, shared_np_array

import cv2
import numpy as np
import array 
from typing import List, Dict, Literal, Tuple, Union, Optional, Any, Callable
from data_collection.data_collection import LoggerSet, Logger
from multiprocessing import Process

from multiprocessing.managers import BaseManager, BaseProxy, SyncManager, ValueProxy

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

        logger.setup_video_saver(resolution=resolution, framerate=framerate)

        presets = dict( main={"size":resolution, "format": "RGB888"}, queue=False, controls={"FrameDurationLimits": (int(1e6/framerate), int(1e6/framerate))}, transform=Transform(hflip=True, vflip=True), buffer_count=1)
        for k, v in config_overrides.items():
            presets[k] = v


        video_config = picam2.create_video_configuration(**presets) # type: ignore
        picam2.configure(video_config) # type: ignore
        picam2.start()


        self.cap = picam2
        self.logger = logger

    def step(self) -> Tuple[Union[None, np.ndarray]]:
        img = self.cap.capture_array("main")[:, :, :3] # type: ignore 

        self.logger.log_time('time')
        self.logger.save_video_frame(img)
        return (img, )

    @classmethod
    def create_shared_outputs(cls, manager:BaseManager)->List[Optional[BaseProxy]]:
        return [None]

    @classmethod
    def create_shared_outputs_rw(cls, manager:BaseManager, resolution: Tuple[int, int]=(0, 0)):
        assert isinstance(manager, SyncManager)

        r, w = shared_np_array(manager, 'B', np.zeros((*resolution, 3), dtype=np.uint8))
        return [r], [w]


    @classmethod
    def entry(
        cls, resolution=0, framerate=0, config_overrides:Any={}, 
        logger_set: Optional[LoggerSet]=None, 
        **kwargs
        ):

        assert isinstance(logger_set, LoggerSet)
        logger = logger_set.get_logger(**kwargs)

        return cls(resolution, framerate, config_overrides, logger)

    @staticmethod
    def proxy_assigner(proxy, value):

        if isinstance(value, np.ndarray):
            proxy[:] = array.array('B',value.ravel())
        else:
            raise NotImplementedError

    @staticmethod
    def get_proxy_reader(dimension):
        # for the receiver
        def proxy_reader(proxy):
            return np.array(proxy[:]).reshape(dimension)

        return proxy_reader


    

    @classmethod
    def create_camera_component(
        cls, 
        init_kwargs: Dict, 
        mainloop: Callable, 
        main_kwargs: Dict, 
        manager: BaseManager, 
    ) -> Tuple[List[Optional[BaseProxy]], Dict[str, BaseProxy], "function", "function"]:

        # messy! 
        resolution = init_kwargs["resolution"]
        n_channel = 3
        array_dim = (*resolution, n_channel)

        def get_ndarray_proxy(manager, array_dim):

            # uint8: https://docs.python.org/3/library/array.html#module-array
            array_length = np.prod(array_dim)

            flatten_array = manager.Array('B', [0]*array_length)

            return flatten_array

        output_proxies = [get_ndarray_proxy(manager, array_dim)]

        proxy_reader = cls.get_proxy_reader(array_dim)
                
        def starter(input_proxies: List[BaseProxy]=[], other_proxies: Dict[str, BaseProxy]={}) -> Process:
            process = Process(
                target=mainloop, 
                kwargs=dict(
                    instantiater = cls.entry,  #bad! 
                    init_kwargs = init_kwargs, 
                    proxy_assigner = cls.proxy_assigner, 
                    proxy_reader = default_proxy_reader,
                    input_proxies = input_proxies, 
                    output_proxies = output_proxies,
                    other_proxies = other_proxies,   
                    **main_kwargs  
                    )
                )
            process.start()
            return process

        return output_proxies, {}, starter, proxy_reader

