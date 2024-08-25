from functools import cache
from math import atanh
from typing import Dict
import torch
tch = torch

import torchaudio as ta
import torchaudio.transforms as tatx
import torchaudio.functional as tafn
from pathlib import Path
import plotly.express as px
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, Dataset, StackDataset
import torchaudio as ta


import scipy.signal as ss

def show_spectrum(sig, rate):
    f, t, sxx = ss.spectrogram(sig, fs=rate)
    return px.imshow(np.log10(sxx), y=f, x=t, aspect='auto' )


@np.vectorize
def tri(idx, length):
    end = length - 1
    if not (idx // end) % 2: 
        return idx % end
    else: 
        return end - (idx % end)

def loop_audio(sig, fs, dur, seg_st, seg_ed):
    hop_length = 200
    win_dur = hop_length/fs

    seg_st = seg_st//win_dur
    seg_ed = seg_ed//win_dur

    spectra = tatx.Spectrogram(power=2, hop_length=hop_length)(sig)

    n_win = dur//win_dur
    idx = tri(np.arange(n_win), seg_ed-seg_st) + seg_st

    return tatx.GriffinLim(hop_length=hop_length)(spectra[..., idx])




class CommonVoiceDataset(Dataset):
    def __init__(self, path, target_fs):
        self.path = Path(path)
        self.clip_path = self.path/'clips'
        self.fs = target_fs

    def use_df(self, df):
        self.df_to_use = df

    def __len__(self):
        return len(self.df_to_use)

    def __getitem__(self, index):
        assert hasattr(self, 'df_to_use')
        fname = self.df_to_use.iloc[index]['path']
        sig = self.load_audio(fname)
        sent = self.df_to_use.iloc[index]['sentence']
        return sig, sent

    @cache
    def get_df(self, df_name):
        matches = list(filter(lambda f:f.name==f'{df_name}.tsv',  self.path.iterdir()))
        assert len(matches)
        return pd.read_csv(matches[0], sep='\t')  

    @cache
    def load_audio(self, fname, suffix='.mp3'):
        waveform, fs = ta.load(self.clip_path/(fname+suffix))
        return tafn.resample(waveform, fs, self.fs)


# dataset transformer
def transform_dataset(ds: Dataset, func):
    """
    not in use yet
    """
    class _cls(Dataset):
        def __len__(self):
            return len(ds) # type: ignore

        def __getitem__(self, index):
            return func(ds[index])
    return _cls()


class NegativeSampleSet(Dataset):
    def __init__(self, ds: CommonVoiceDataset, duration_ms):
        self.ds = ds
        self.duration_ms = duration_ms
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return get_random_segment(self.ds, index, self.duration_ms), tch.tensor(0.0)

class PositiveSampleSet(Dataset):
    def __init__(self, ds: CommonVoiceDataset, duration_ms, seg_df, jitter_pct):
        self.ds = ds
        self.duration_ms = duration_ms
        self.seg_df = seg_df
        self.jitter_pct = jitter_pct

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return get_middle_segment_jitter(self.ds, self.seg_df, index, self.duration_ms, jitter_pct=self.jitter_pct), tch.tensor(1.0)


class MergePosNegSet(Dataset):
    def __init__(self, pos_ds, neg_ds):
        self.pos_ds = pos_ds 
        self.neg_ds = neg_ds

        self.pos_ds_len = len(self.pos_ds)
        self.neg_ds_len = len(self.pos_ds)

    def __len__(self):
        return self.pos_ds_len*2
        

    def __getitem__(self, index):
        if (index >= self.pos_ds_len*2) or (index < 0) : 
            raise IndexError
        
        if index < self.pos_ds_len:
            return self.pos_ds[index]

        else:
            idx = np.random.randint(self.neg_ds_len)
            return self.neg_ds[idx]
        





def get_random_segment(ds, index, duration_ms):
    sig, sent = ds[index]

    seg_size = round(duration_ms*ds.fs/1000)
    start_max = sig.shape[-1] - seg_size
    start = np.random.randint(start_max)
    return sig[..., start: start+seg_size]

def get_middle_segment_jitter(ds, seg_df, index, duration_ms, jitter_pct=0.2):
    sig, sent = ds[index]
    
    seg_row = seg_df.iloc[index]
    mid = round((seg_row.start+seg_row.end)/2)
    
    seg_size = round(duration_ms*ds.fs/1000)


    max_shift = round(jitter_pct*seg_size/2)
    shift = np.random.randint(max_shift*2)-max_shift
    # TODO: can still go out of range at the end
    start = max(0, mid-seg_size//2+shift)

    return sig[..., start:start+seg_size]



def get_middle_segment(ds, seg_df, index, duration_ms):
    sig, sent = ds[index]
    
    seg_row = seg_df.iloc[index]
    mid = round((seg_row.start+seg_row.end)/2)
    oneside = int(duration_ms*ds.fs/1000/2)

    #     |pad| we |pad|
    # 0000....111111....0000
    return sig[..., mid-oneside:mid+oneside]

    
def get_segment(ds: CommonVoiceDataset, seg_df, index): 

    sig, sent = ds[index]

    seg_row = seg_df.iloc[index]

    return sig[..., int(seg_row.start):int(seg_row.end)]
    