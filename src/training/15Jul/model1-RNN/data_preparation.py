from __future__ import annotations

import sys
from typing_extensions import deprecated
sys.path += ['..', '../..', '../../..']


from data_collection.data_collection import LoggerSet, Logger

import numpy as np
import pandas as pd
import plotly.express as px
from data_collection.video_data import get_frame_iterator
from pathlib import Path
from typing import Callable, Iterable, Tuple, List, Any
from tqdm import tqdm
import datetime
import tensorflow as tf
import tensorflow.keras as keras # type: ignore 



def mean_nominmax(x):
    return (x.sum()-x.max()-x.min())/(len(x)-2)

def data_prep(obj: Session):

    def add_col_rolling_mean_nominmax(window, shift, center):
        df = obj.data['angular_speed_control_df']
        df['angular_velocity_smooth'] = df['angular_velocity'].rolling(window, center=center).apply(mean_nominmax).shift(shift)

    add_col_rolling_mean_nominmax(window=50, shift=-25, center=False )


    def add_angular_velocity_smooth_offset_columns():
        adf = obj.data['angular_speed_control_df']
        cdf = obj.data['camera_df']
                
        offset_list = [50, 100, 200, 400]
        tidxs = [
            match_time(adf['time_AngularSpeedControl'], cdf['time'],  t2_offset_ms=_offset)
            for _offset in offset_list
        ]
        col_names: List = ['angular_velocity_smooth']
        omega_shifted = np.concatenate([
                adf[col_names].values[_tidx] 
                for _tidx in tidxs
            ], axis=1)

        column_names = [c+'_'+str(t) for t in offset_list for c in col_names]
        cdf[column_names] = omega_shifted
        obj.data['add_angular_velocity_smooth_offset_columns'] = column_names

    add_angular_velocity_smooth_offset_columns()


    def add_speed_column():
        adf = obj.data['angular_speed_control_df']
        cdf = obj.data['camera_df']

        tidx = match_time(cdf['time'],  adf['time_AngularSpeedControl'])
        adf['frame_match'] = tidx
        mean_speed = adf.groupby('frame_match')['speed'].mean()

        # this is the mean speed after the end (acquisition) of the frame until the end of next frame
        cdf['mean_speed'] = mean_speed

    add_speed_column()

    

def sample_prep(obj: Session):
    frames, camera_df = obj.data['frames'], obj.data['camera_df']

    camera_df = camera_df.query('mean_speed>0')
    out = camera_df[obj.data['add_angular_velocity_smooth_offset_columns']]
    
    frames = frames[out.index]

    obj.samples = frames, out.values/100
    


def stack_samples(objs: List[Session], sample_axis=0):
    samples_stacked = []
    for s in zip(*[o.samples for o in objs]):
        samples_stacked.append(np.concatenate(s, axis=sample_axis))
    return samples_stacked



def match_time(t1_col, t2_col, t1_offset_ms=0, t2_offset_ms=0, sort_check=True):
    """
    return the indices of t1_col that match t2
    """
    if sort_check: 
        sortedt1 = np.sort(t1_col)
        sortedt2 = np.sort(t2_col)
        sortedt1 = sortedt1[~np.isnan(sortedt1)]
        sortedt2 = sortedt2[~np.isnan(sortedt2)]

        assert np.all(t1_col[~np.isnan(t1_col)] == sortedt1)
        assert np.all(t2_col[~np.isnan(t2_col)] == sortedt2)
    return np.searchsorted(t1_col+pd.to_timedelta(t1_offset_ms, unit='ms'), t2_col+pd.to_timedelta(t2_offset_ms, unit='ms'))



# def prepare_parquet_for_tf(logpath, name, col_names:List=['out']):
# NOT WORKING
#     logpath = Path(logpath)
#     logger = Logger(logpath, name, overwrite_ok=True) 

#     df = logger.load_as_df()

#     for c in col_names: 
#         df[c] = df[c].dropna().apply(lambda x: x.numpy())

#     df.to_parquet(logpath/f"{name}.parquet")

    
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
    def __init__(
        self, 
        logpath, 
        camera_log_name = 'Picamera2V2', 
        angular_speed_control_log_name='AngularSpeedControlV3'
    
    ):
        logpath = Path(logpath)

        errs = {}
        if not (logpath/f'{camera_log_name}.parquet').exists():
            errs = prepare_parquets(logpath)

        self.camera_log_name = camera_log_name
        self.angular_speed_control_log_name = angular_speed_control_log_name

        self.logger_set = LoggerSet(logpath, overwrite_ok=False) 

        self.data = {}
        self.errs = errs
        self.logpath = logpath
        self.samples: Any


        self.load_data()


    # def main(self):
    #     # main
    #     
    #     self.add_col_lin_acc('angular_speed_control_df', 'linear_acc')
    #     self.add_col_sin_cos('angular_speed_control_df', 'angle')
    #     self.add_col_frame_idx('angular_speed_control_df', 'camera_df', time_offset_ms=100)
    #     self.add_col_rolling_mean_nominmax(
    #         'angular_speed_control_df', 
    #         'angular_velocity', 'angular_velocity_smooth',
    #         window=50, shift=-49, center=False 
    #         )

    #     self.add_col_rolling_mean_nominmax(
    #         'angular_speed_control_df', 
    #         'left', 'left_smooth',
    #         window=50, shift=-49, center=False 
    #         )
    #     self.add_col_rolling_mean_nominmax(
    #         'angular_speed_control_df', 
    #         'right', 'right_smooth',
    #         window=50, shift=-49, center=False 
    #         )

    def load_data(self):
        """
        reusable
        """
        self.data['camera_df'] =  pd.read_parquet(self.logpath/f'{self.camera_log_name}.parquet')
        self.data['angular_speed_control_df'] = pd.read_parquet(self.logpath/f'{self.angular_speed_control_log_name}.parquet')
        self.data['frames'] = stack_frames(get_frame_iterator(self.logpath/f"{self.camera_log_name}/video"))


        def load_FrameMLToAngularVelocity(): 
            try: 
                df = self.logger_set.load_logger('FrameMLToAngularVelocity').load_as_df()
                df['out'] = df['out'].dropna().apply(lambda x: x.numpy())
            
                self.data['FrameMLToAngularVelocity'] = df 
            except: 
                print('FrameMLToAngularVelocity not found')
                pass 
        load_FrameMLToAngularVelocity()

        def trim_frame_df():
            df_len = len(self.data['camera_df'])
            frame_len = len(self.data['frames'])
            min_len = min(df_len, frame_len)

            self.data['camera_df'] = self.data['camera_df'][:min_len]
            self.data['frames'] = self.data['frames'][:min_len]
            
        trim_frame_df()


    @deprecated('bad code')
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

    @deprecated('bad code')
    def add_col_sin_cos(self, df_name, column):
        df = self.data[df_name]

        df[column+'_sin'] = np.sin(df[column]/180*np.pi)
        df[column+'_cos'] = np.cos(df[column]/180*np.pi)


    @deprecated('bad code')
    def add_col_rolling_mean_nominmax(self, df_name, column, new_col, window, shift, **kwargs):
        def mean_nominmax(x):
            return (x.sum()-x.max()-x.min())/(len(x)-2)
        df = self.data[df_name]
        assert not new_col in df.columns
        df[new_col] = df[column].rolling(window, **kwargs).apply(mean_nominmax).shift(shift)

    @deprecated('bad code')
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
    def load_multiple_session(cls, parent_folder, **kwargs):
        """
        reusable - instantiates only
        """
        results = []
        for d in Path(parent_folder).iterdir():
            results.append(cls(d, **kwargs))

        return results

    # @deprecated('this is a very bad idea. just prepare the sample for each session separately. ')
    # @staticmethod
    # def concatenate_multiple_sessions(sessions: List["Session"], df_name='angular_speed_control_df'):
    #     """
    #     MODIFY THE DATAFRAME *IN PLACE* as a side effect

    #     depends on match_frame_idx
    #     """

    #     frame_set_list = []
    #     df_list = []
    #     total_frames = 0

    #     for s in sessions:

    #         frames, df = s.data['frames'], s.data[df_name]
    #         df['offset_total_frame_idx'] = df['offset_frame_idx'] + total_frames
    #         df['exact_total_frame_idx'] = df['exact_frame_idx'] + total_frames

    #         total_frames += len(frames)
    #         df_list.append(df)
    #         frame_set_list.append(frames)

    #     return np.concatenate(frame_set_list, axis=0), pd.concat(df_list)
