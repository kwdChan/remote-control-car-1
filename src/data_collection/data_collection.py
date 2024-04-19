from pathlib import Path
from typing import List, Dict, Literal, Tuple, Union, cast
import datetime, time 
import numpy as np 
import pandas as pd
import pickle
from .video_data import VideoSaver
from tqdm import tqdm
from copy import copy
class LoggerSet:
    def __init__(self, path='../log/temp', overwrite_ok=False, override_save_interval=None):
        path_obj = Path(path) 
        
        # if path_obj.exists():
        #     assert overwrite_ok
    
        path_obj.mkdir(parents=True, exist_ok=True)

        self.override_save_interval = override_save_interval
        self.path = path_obj
        self.overwrite_ok = overwrite_ok

    def get_logger(self, name, save_interval=30):
        if not self.override_save_interval is None:
            save_interval = self.override_save_interval

        return Logger(self.path, name, save_interval=save_interval, overwrite_ok=self.overwrite_ok)

    def load_logger(self, name):
        match = [ l for l in self.get_all_logger() if l.name == name ]
        if match:
            return match[0]
        else:
            raise KeyError 
        
    
    def get_all_logger(self):
        result = []
        for d in self.path.iterdir():
            if d.is_dir():
                result.append(Logger(self.path, name=d.stem, overwrite_ok=True))
        return result

    def export_to_parquet(self, **kwargs):
        failed = {}
        for logger in tqdm(self.get_all_logger()):
            try: 
                df = logger.load_as_df(**kwargs)
                df.to_parquet(self.path/f"{logger.name}.parquet")
            except Exception as e:
                failed[logger.name] = e

        if len(failed):
            print('some failed')
        return failed
            
            
class Logger:

    def __init__(self, path: Union[Path, str] ='../log/temp', name='unnamed', save_interval:float=30, overwrite_ok=False):
        """
        a shared logger if all control components 

        path is a directory 

        chucks of data
        """
        path_obj = Path(path) / name
        if path_obj.exists():
            assert overwrite_ok
        path_obj.mkdir(parents=True, exist_ok=overwrite_ok)


        self.path = path_obj
        self.save_interval = save_interval
        self.name = name
        self.video_saver: Union[None, 'VideoSaver'] = None
        
        # states
        self.records = []
        self.chuck_number = 0
        self.idx = 0
        self.last_saved = time.monotonic()

    def __repr__(self):
        return f"Logger: {self.name}"

    def set_idx(self, idx):     
        self.idx = idx 
    
    def increment_idx(self):
        self.idx += 1

    def log_time(self, key='time'):
        self.log(key, datetime.datetime.now())

    def log(self, key, value):

        """
        log the value and the current idx
        ideally all values of the same key should have the same structure
        """

        new_record = dict(idx=self.idx, key=key, value=copy(value))
        self.records.append(new_record)

        if (time.monotonic() - self.last_saved) >= self.save_interval:
            self.save()

    def save(self):
        
        with open(self.path/f"{self.chuck_number}.pkl", "wb") as fp:
            pickle.dump(self.records, fp, pickle.HIGHEST_PROTOCOL)
        self.records = []
        self.chuck_number += 1
        self.last_saved = time.monotonic()

    def setup_video_saver(self, **kwargs):
        self.video_saver = VideoSaver(self.path/'video', **kwargs)

    def save_video_frame(self, arr):
        vidx, fidx, monotonic = cast(VideoSaver, self.video_saver).save_frame(arr) 
        self.log('vidx', vidx)
        self.log('fidx', fidx)
        self.log('monotonic', monotonic)
        
    def load(self, min_idx=0, max_idx=999):
        data = []
        
        for idx in range(min_idx, max_idx+1):
            path = self.path/f"{idx}.pkl"
            if not path.exists(): 
                continue
            
            with open(path, 'rb') as fp:
                data.append(pickle.load(fp))

        return sum(data,  start=[])

    def load_as_df(self, min_idx=0, max_idx=999):

        result = self.load(min_idx, max_idx)
        result = pd.DataFrame(result)
        return result.pivot(index='idx', columns=['key']).droplevel(0, 1)

