

import sys
sys.path += ['..']

from data_collection.data_collection import LoggerSet, Logger

import numpy as np
import pandas as pd
import plotly.express as px
from data_collection.video_data import get_frame_iterator
from pathlib import Path
from typing import Callable, Iterable, Tuple, List
from tqdm import tqdm
import datetime
import tensorflow as tf
import tensorflow.keras as keras # type: ignore 

def prepare_parquets(logpath):
    logpath = Path(logpath)
    logger_set = LoggerSet(logpath, overwrite_ok=False) 
    return logger_set.export_to_parquet()

def stack_frames(gen: Iterable[Tuple[int, np.ndarray]]):
    """
    from get_frame_iterator
    """
    frames = []
    for idx, img in gen:
        frames.append(img)
    return np.stack(frames, axis=0)

class Session:
    def __init__(self, logpath):
        logpath = Path(logpath)

        errs = {}
        if not (logpath/'PicameraV2.parquet').exists():
            errs = prepare_parquets(logpath)

        self.data = {}
        self.errs = errs
        self.logpath = logpath

        # main
        self.load_data()
        self.add_col_lin_acc('angular_speed_control_df', 'linear_acc')
        self.add_col_sin_cos('angular_speed_control_df', 'angle')
        self.add_col_frame_idx('angular_speed_control_df', 'camera_df', time_offset_ms=100)
        self.add_col_rolling_mean_nominmax(
            'angular_speed_control_df', 
            'angular_velocity', 'angular_velocity_smooth',
            window=50, shift=-49, center=False 
            )

        self.add_col_rolling_mean_nominmax(
            'angular_speed_control_df', 
            'left', 'left_smooth',
            window=50, shift=-49, center=False 
            )
        self.add_col_rolling_mean_nominmax(
            'angular_speed_control_df', 
            'right', 'right_smooth',
            window=50, shift=-49, center=False 
            )

    def load_data(self):
        """
        reusable
        """
        self.data['camera_df'] =  pd.read_parquet(self.logpath/'PicameraV2.parquet')
        self.data['angular_speed_control_df'] = pd.read_parquet(self.logpath/'AngularSpeedControlV2.parquet')
        self.data['frames'] = stack_frames(get_frame_iterator(self.logpath/"PicameraV2/video"))

    def add_col_lin_acc(self, df_name, column_name): 
        """
        """
        
        def to_multiple_columns(x):
            return x.iloc[0]

        df = self.data[df_name].copy()
        expanded = df[[column_name]].dropna().apply(to_multiple_columns, axis=1, result_type='expand')
        expanded.columns=['linacc0', 'linacc1', 'linacc2']

        df = pd.merge(df, expanded, on='idx', how='inner')

        self.data[df_name] = df

    def add_col_sin_cos(self, df_name, column):
        df = self.data[df_name]

        df[column+'_sin'] = np.sin(df[column]/180*np.pi)
        df[column+'_cos'] = np.cos(df[column]/180*np.pi)




    def add_col_rolling_mean_nominmax(self, df_name, column, new_col, window, shift, **kwargs):
        def mean_nominmax(x):
            return (x.sum()-x.max()-x.min())/(len(x)-2)
        df = self.data[df_name]
        assert not new_col in df.columns
        df[new_col] = df[column].rolling(window, **kwargs).apply(mean_nominmax).shift(shift)



    def add_col_frame_idx(self, df_name,  camera_df_name, time_offset_ms=100):
        """
        """
        camera_df = self.data[camera_df_name].copy()
        angular_speed_control_df = self.data[df_name].copy()
            
        camera_df['time_expected'] = camera_df['time'] + pd.to_timedelta(time_offset_ms, unit='ms')
        angular_speed_control_df['offset_frame_idx'] = camera_df['time_expected'].searchsorted(angular_speed_control_df['time_AngularSpeedControl'], side='right')

        camera_df['time_exact'] = camera_df['time']
        angular_speed_control_df['exact_frame_idx'] = camera_df['time_exact'].searchsorted(angular_speed_control_df['time_AngularSpeedControl'], side='left')

        self.data[df_name] = angular_speed_control_df

    def __repr__(self):
        return f"Session('{str(self.logpath)}')"

    @classmethod
    def load_multiple_session(cls, parent_folder, inclusion_filter=lambda d: d.stem!='excluded'):
        """
        reusable - instantiates only
        """
        results = []
        for d in filter(inclusion_filter, Path(parent_folder).iterdir()):
            results.append(cls(d))

        return results

    @staticmethod
    def concatenate_multiple_sessions(sessions: List["Session"], df_name='angular_speed_control_df'):
        """
        MODIFY THE DATAFRAME *IN PLACE* as a side effect

        depends on match_frame_idx
        """

        frame_set_list = []
        df_list = []
        total_frames = 0

        for s in sessions:

            frames, df = s.data['frames'], s.data[df_name]
            df['offset_total_frame_idx'] = df['offset_frame_idx'] + total_frames
            df['exact_total_frame_idx'] = df['exact_frame_idx'] + total_frames

            total_frames += len(frames)
            df_list.append(df)
            frame_set_list.append(frames)

        return np.concatenate(frame_set_list, axis=0), pd.concat(df_list)
