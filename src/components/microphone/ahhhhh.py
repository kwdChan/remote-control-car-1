from typing import Callable, List, Union
from typing_extensions import deprecated
import numpy as np

from data_collection.data_collection import LoggerSet, Logger
from .viz import get_table, get_power_plot
from .microphone import MicrophoneReader
from plotly.graph_objects import FigureWidget

from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from scipy.fft import fft, fftfreq



class AhhhhhWheelController:
    def __init__(self, microphone: MicrophoneReader, logger: Logger):
        self.ahhhhh_detector = AhhhhhDetector(microphone, logger)
        self.logger = logger

        # state 
        self.pitch_hist = [np.nan, np.nan, np.nan]
        self.is_valid_hist = [False, False]
        

    def step(self):
        """
        TODO: 
        """
        pitch_power, basefreq, pitch = self.ahhhhh_detector.step()

        self.pitch_hist = self.pitch_hist[1:] + [pitch]

        max_diff = 0
        for i in range(len(self.pitch_hist)-1):
            for j in range(i, len(self.pitch_hist)):
                max_diff = np.nanmax([max_diff, angle_deg_between(self.pitch_hist[i], self.pitch_hist[j])])
        
        this_step_valid = (max_diff<60) and (pitch_power>375)
        
    
        if this_step_valid and all(self.is_valid_hist):            
            speed = 60
        else: 
            speed = 0

        self.is_valid_hist = self.is_valid_hist[1:] + [this_step_valid]

        # use np.nanmedian(self.pitch_hist) so that it is more resilient to rare nan
        balance = angle2proportion_v2(np.nanmedian(self.pitch_hist), 220, 120, 0.3)

        self.logger.log_time("AhhhhhWheelController")
        self.logger.log('speed_or_invalid', speed)
        self.logger.log('median_pitch', np.nanmedian(self.pitch_hist))
        self.logger.log('balance', balance)

        return balance*speed, (1-balance)*speed



    @staticmethod
    def main(index, sender: Connection, frame_length, signal_nframe, sxx_nframe, logger_set: LoggerSet, name:str, **kwargs):
        """
        for a sample rate of  16000, use
            frame_length: 800
            signal_nframe: 5
            sxx_nframe: 5 (much less important)
        """

        logger = logger_set.get_logger(name=name, **kwargs)
        mic = MicrophoneReader(index, frame_length, signal_nframe, sxx_nframe)
        component = AhhhhhWheelController(mic, logger)
        while True:
            logger.increment_idx()
            result  = component.step()
            sender.send((result, name, logger.idx))

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
    def __init__(self, micophone: MicrophoneReader, logger: Logger):
        self.micophone = micophone
        self.logger = logger

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
        self.micophone.sample()

        latest_sxx = self.micophone.sxx_buffer[-1]

        basefreq, pitch_power = find_pitch_with_fft(latest_sxx,  self.micophone.freqs, freq_range=(90, 3000) )

        pitch = freq_to_pitch(basefreq) 

        self.logger.log_time("AhhhhhDetector")
        self.logger.log("latest_sxx", latest_sxx)
        self.logger.log("basefreq", basefreq)
        self.logger.log("pitch_power", pitch_power)
        self.logger.log("pitch", pitch)
        
        return pitch_power, basefreq, pitch


def angle_deg_between(a1, a2):
    return min((a1-a2)%360, (a2-a1)%360)

@deprecated("use angle2proportion_v2")
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


def angle2proportion_v2(angle_deg, centre, pitch_scale_oneside, balance_scale_oneside):
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

    proportion = angle_centred/(pitch_scale_oneside/balance_scale_oneside) + 0.5
    proportion = min(proportion, 0.5+balance_scale_oneside)
    proportion = max(proportion, 0.5-balance_scale_oneside)
    return proportion


def rolling_mean(arr, kernel=np.ones(3)/3): 
    return np.correlate(arr, kernel, mode='same')
    
def adjust(sxx, bandwidth=45):
    return sxx - rolling_mean(sxx, np.ones(bandwidth)/bandwidth) 


def fft_power_spectra(sxx, freqs):
    sxx = sxx-sxx.mean()
    y = fft(sxx)
    freqs = fftfreq(len(freqs), freqs[1]-freqs[0])

    pitch_power = y.real**2+y.imag**2 # type: ignore
    return 1/freqs[:len(freqs)//2], pitch_power[:len(freqs)//2]


def fft_power_spectra_masked_by_adjusted(sxx, freqs):
    f, adjusted_sxx = fft_power_spectra(rolling_mean(adjust(sxx, bandwidth=20)), freqs)
    f, sxx  = fft_power_spectra(sxx, freqs)

    # TODO: constant
    return f, sxx*(adjusted_sxx>(20e3)) 


def find_pitch_with_fft(sxx, freqs, freq_range=(90, 600)):
    freq_pitch, sxx_pitch = fft_power_spectra_masked_by_adjusted(sxx, freqs)
    bidx = (freq_pitch>freq_range[0]) & (freq_pitch<freq_range[1])
    argmax = np.argmax(sxx_pitch[bidx])

    return freq_pitch[bidx][argmax], sxx_pitch[bidx][argmax]


def freq_to_pitch(freq):
    return (np.log2(freq) % 1) * 360
