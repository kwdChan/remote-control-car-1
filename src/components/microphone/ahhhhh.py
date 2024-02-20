from typing import Callable, List, Union
import numpy as np
from .viz import get_table, get_power_plot
from .microphone import MicrophoneReader
from plotly.graph_objects import FigureWidget

from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection



class AhhhhhWheelController:
    def __init__(self, microphone: MicrophoneReader):
        self.ahhhhh_detector = AhhhhhDetector(microphone)

        # state 
        self.pitch = [np.nan, np.nan, np.nan]
        self.last_step_valid = False
    def step(self):
        """
        TODO: 
        """
        latest_sxx, detected_freqs, strengths, basefreq, pitch = self.ahhhhh_detector.step()

        self.pitch = self.pitch[1:] + [pitch]

        max_diff = np.nanmax(self.pitch) - np.nanmin(self.pitch)

        this_step_valid = (max_diff<75) and (latest_sxx[1:].mean()>20)
    
        if this_step_valid and self.last_step_valid:            
            speed = 80
        else: 
            speed = 0

        self.last_step_valid = this_step_valid

        balance = (np.nanmedian(pitch) - 180)/360 + 0.5
        balance = max(0, balance)
        balance = min(1, balance)
        
        return balance*speed, (1-balance)*speed



    @staticmethod
    def main(index, sender: Connection, frame_length=800, signal_nframe=5, sxx_nframe=5):
        mic = MicrophoneReader(index, frame_length, signal_nframe, sxx_nframe)
        component = AhhhhhWheelController(mic)
        while True:
            result  = component.step()
            sender.send(result)

    @staticmethod
    def start(index, frame_length=800, signal_nframe=5, sxx_nframe=5):
        receiver, sender = Pipe(False)
        
        p = Process(target=AhhhhhWheelController.main, args=(index, sender, frame_length, signal_nframe, sxx_nframe))
        p.start()
        return p, receiver
        



class AhhhhhDetector:
    def __init__(self, micophone: MicrophoneReader):
        self.micophone = micophone

        # state
        # visualisation objects 
        self.slice_fig: Union[None, FigureWidget] = None
        self.slice_update: Union[None, Callable] = None

        self.table: Union[None, FigureWidget] = None
        self.table_update: Union[None, Callable] = None

    def setup_power_viz(self):
        slice_fig, slice_update = get_power_plot(self.micophone.freqs)
        self.slice_fig = slice_fig
        self.slice_update = slice_update
        return slice_fig

    def setup_table_viz(self):
        table, table_update = get_table(['frequency', 'strength'])
        self.table = table
        self.table_update = table_update
        return table

    def step(self):

        mic = self.micophone 
        freqs = mic.freqs

        mic.sample()

        latest_sxx = mic.sxx_buffer[-1]
        latest_sxx = latest_sxx[np.where(freqs < 2000) ]

        # normalise againist other frequencies (not time)
        latest_sxx_norm = normalise(latest_sxx)

        # the strengths are relative to other frequencies (not time)
        detected_freqs, strengths = get_freq_bands(latest_sxx_norm, freqs, 2, 4)

        basefreq = find_basefreq(np.array(detected_freqs))  

        pitch = freq_to_pitch(basefreq) 
        
        return latest_sxx, detected_freqs, strengths, basefreq, pitch


    def loop(self, callbacks: List[Callable], table_update_interval=0, power_update_interval=0):
        """
        for notebook use only
        don't do callback chain
        """
        idx = 0
        while True:
            latest_sxx, detected_freqs, strengths, basefreq, pitch = self.step()
            
            for cb in callbacks:
                cb(latest_sxx, detected_freqs, strengths, basefreq, pitch)

            if table_update_interval and (not idx % table_update_interval):
                assert self.table_update, 'table graph not set up yet'
                self.table_update(dict(
                    frequency = detected_freqs, 
                    strength = strengths, 
                ))

            if power_update_interval and (not idx % power_update_interval):
                assert self.slice_update, 'power power no set up yet'
                self.slice_update(latest_sxx)
                
            idx += 1






def get_freq_bands(values, freqs, threshold, peak_threshold):

    values = np.array(values)
    freqs = np.array(freqs)

    bands = [[]]
    for idx, v in enumerate(values):
        if v > threshold:
            bands[-1].append(idx)
        elif len(bands[-1]):
            bands.append([])

    bands = [i for i in bands if len(i)>0 and len(i)<80]

    freq_centres = []
    freq_peak_powers = []
    for each_band in bands:
        freq_peak_power = values[each_band].max()
        if freq_peak_power < peak_threshold:
            continue
        

        centre_idx_idx = np.argmax(values[each_band])
        centre_idx = each_band[centre_idx_idx]
        freq_centre = freqs[centre_idx]
        

        freq_centres.append(freq_centre)
        freq_peak_powers.append(freq_peak_power)

    return freq_centres, freq_peak_powers



def normalise(values):
    median = np.median(values)
    mad = np.median(abs(values-median))

    return (values - median)/mad



def find_peaks(normalised, freqs):


    freq_centres, freq_peak_powers = get_freq_bands(normalised, freqs, 2, 4)

    #pitches = [freq_to_pitch(f) for f in freq_centres]

    
    freq_centres = np.array([0]+[f for f in freq_centres if f > 40])

    basefreq = np.nan
    if len(freq_centres)>3:
        gap1 = freq_centres[1:] - freq_centres[:-1]
        #gap2 = freq_centres[2:] - freq_centres[:-2]

        basefreq = np.median(np.concatenate([gap1]))
        

    return basefreq, freq_centres, freq_peak_powers


def freq_to_pitch(freq):
    return (np.log2(freq) % 1) * 360



def find_basefreq(freqs):

    freqs = [0] + [f for f in freqs if f > 40]
    freqs = np.array(freqs)
    basefreq = np.nan

    if len(freqs)>3:
        gap1 = freqs[1:] - freqs[:-1]
        #gap2 = freqs[2:] - freqs[:-2]

        gaps = np.sort(np.concatenate([gap1]))


        basefreq = gaps[(len(gaps)+1)//2]
        
    return basefreq