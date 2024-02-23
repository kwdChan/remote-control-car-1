from pathlib import Path
from typing import List, Dict, Literal, Union
import datetime, time 
import numpy as np 
import pandas as pd
import pickle

class LoggerSet:
    def __init__(self, path='../log/temp', overwrite_ok=False):
        path_obj = Path(path) 
        
        # if path_obj.exists():
        #     assert overwrite_ok
    
        path_obj.mkdir(parents=True, exist_ok=True)

        
        self.path = path_obj
        self.overwrite_ok = overwrite_ok

    def get_logger(self, name, save_interval=30):
        return Logger(self.path, name, save_interval=save_interval, overwrite_ok=self.overwrite_ok)

    def get_all_logger(self):
        result = []
        for d in self.path.iterdir():
            if d.is_dir():
                result.append(Logger(self.path, name=d.stem, overwrite_ok=True))
        return result
            
    

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

        new_record = dict(idx=self.idx, key=key, value=value)
        self.records.append(new_record)

        if (time.monotonic() - self.last_saved) >= self.save_interval:
            self.save()

    def save(self):
        
        with open(self.path/f"{self.chuck_number}.pkl", "wb") as fp:
            pickle.dump(self.records, fp, pickle.HIGHEST_PROTOCOL)
        self.records = []
        self.chuck_number += 1
        self.last_saved = time.monotonic()


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


    def convert_camera_data_for_transfer(self, img_key, min_idx=0, max_idx=9999,):
        
        raise NotImplementedError()
        for idx in range(min_idx, max_idx+1):
            path = self.path/f"{idx}.pkl"
            if not path.exists(): 
                continue

        pass
                    
