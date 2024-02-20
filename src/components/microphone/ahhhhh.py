from typing import Callable, List, Union
from typing_extensions import deprecated
import numpy as np

from data_collection import Logger, LoggerSet
from .viz import get_table, get_power_plot
from .microphone import MicrophoneReader
from plotly.graph_objects import FigureWidget

from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection



class AhhhhhWheelController:
    def __init__(self, microphone: MicrophoneReader, logger: Logger):
        self.ahhhhh_detector = AhhhhhDetector(microphone)
        self.logger = logger

        # state 
        self.pitch_hist = [np.nan, np.nan, np.nan]
        self.is_valid_hist = [False, False]
        

    def step(self):
        """
        TODO: 
        """
        latest_sxx, detected_freqs, strengths, basefreq, pitch = self.ahhhhh_detector.step()

        self.pitch_hist = self.pitch_hist[1:] + [pitch]


        max_diff = 0
        for i in range(len(self.pitch_hist)-1):
            for j in range(i, len(self.pitch_hist)):
                max_diff = np.nanmax([max_diff, angle_deg_between(self.pitch_hist[i], self.pitch_hist[j])])
        
        this_step_valid = (max_diff<60) and (latest_sxx[1:].mean()>20)
        
    
        if this_step_valid and all(self.is_valid_hist):            
            speed = 80
        else: 
            speed = 0

        self.is_valid_hist = self.is_valid_hist[1:] + [this_step_valid]

        # use np.nanmedian(self.pitch_hist) so that it is more resilient to rare nan
        balance = angle2proportion(np.nanmedian(self.pitch_hist), 220, 60)


        self.logger.log_time()
        self.logger.log('pitch', pitch)
        self.logger.log('basefreq', basefreq)
        self.logger.log('latest_sxx', latest_sxx)
        self.logger.log('detected_freqs', detected_freqs)
        self.logger.log('speed_or_invalid', speed)
        self.logger.log('median pitch', np.nanmedian(self.pitch_hist))
        self.logger.log('balance', balance)


        return balance*speed, (1-balance)*speed



    @staticmethod
    def main(index, sender: Connection, frame_length, signal_nframe, sxx_nframe, logger_set: LoggerSet, **kwargs):
        """
        for a sample rate of  16000, use
            frame_length: 800
            signal_nframe: 5
            sxx_nframe: 5 (much less important)
        """

        logger = logger_set.get_logger(**kwargs)
        mic = MicrophoneReader(index, frame_length, signal_nframe, sxx_nframe)
        component = AhhhhhWheelController(mic, logger)
        while True:
            logger.increment_idx()
            result  = component.step()
            sender.send(result)

    @staticmethod
    def start(index, frame_length, signal_nframe, sxx_nframe, logger_set: LoggerSet, **kwargs):
        """
        for a sample rate of  16000, use
            frame_length: 800
            signal_nframe: 5
            sxx_nframe: 5 (much less important)
        """
        receiver, sender = Pipe(False)
        
        p = Process(target=AhhhhhWheelController.main, args=(index, sender, frame_length, signal_nframe, sxx_nframe, logger_set), kwargs=kwargs)
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
        latest_sxx = latest_sxx#[np.where(freqs < 2000)]
        

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

def angle_deg_between(a1, a2):
    return min((a1-a2)%360, (a2-a1)%360)

def angle2proportion(angle_deg, centre, scale_oneside):
    """
    scale_oneside: (0, 180]

    mapped an angle to a number 
        centre + scale_oneside -> 1
        centre + 0             -> 0.5
        centre - scale_oneside -> 0
    """
    angle_centred = (angle_deg - centre) % 360
    if angle_centred > 180:
        angle_centred = angle_centred - 360

    # angle_centred: (-180, 180]

    proportion = angle_centred/(scale_oneside*2) + 0.5
    proportion = min(proportion, 1)
    proportion = max(proportion, 0)
    return proportion

def total_pitch_power(arr, fundamental_idx, bandwidth_idx=1, n_harmonics=4):
    indices = np.arange(1, n_harmonics+1, dtype=int)*fundamental_idx

    total_power = 0
    for i in indices:
        total_power += arr[i-bandwidth_idx: i+bandwidth_idx+1].sum()
    return total_power




@deprecated("use total_pitch_power")
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


@deprecated("use total_pitch_power")
def normalise(values):
    median = np.median(values)
    mad = np.median(abs(values-median))

    return (values - median)/mad


@deprecated("use total_pitch_power")
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


@deprecated("use total_pitch_power")
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