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
        balance = angle2proportion(np.nanmedian(self.pitch_hist), 220, 120)

        self.logger.log_time("AhhhhhWheelController")
        self.logger.log('speed_or_invalid', speed)
        self.logger.log('median_pitch', np.nanmedian(self.pitch_hist))
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


def find_pitch(sxx, freqs, freq_range=(90, 3000), idx_step=1, n_harmonics=5):
    freq_of_power, powers = get_total_pitch_powers(
        sxx, freqs, freq_range=freq_range, idx_step=idx_step, n_harmonics=n_harmonics
        )
    argmax = np.argmax(powers)
    return freq_of_power[argmax], powers[argmax]
    

def get_total_pitch_powers(sxx, freqs, freq_range=(90, 3000), idx_step=1, n_harmonics=5):
    idx_start = np.where(freqs < freq_range[0])[-1][-1]
    idx_end = np.where(freqs < freq_range[-1])[-1][-1]

    freq_of_power = []
    powers = []
    for freq_idx in range(idx_start, idx_end, idx_step):
        power_ = total_pitch_power(sxx, freq_idx, bandwidth_idx=1, n_harmonics=n_harmonics)
        powers.append(power_)
        freq_of_power.append(freqs[freq_idx])
    
    return freq_of_power, powers


def total_pitch_power(sxx, fundamental_idx, bandwidth_idx=1, n_harmonics=5):
    """
    TODO: index out of range at high frequency
    """
    nth_harmonics = np.arange(1, n_harmonics+1, dtype=int)
    total_power = 0
    n = 1
    for n in nth_harmonics:
        i = n*fundamental_idx
        if (i+bandwidth_idx) >= len(sxx):
            break
        total_power += sxx[i-bandwidth_idx: i+bandwidth_idx+1].mean()
    return total_power/(n-1)

def get_total_pitch_powers_adjusted(sxx, freqs, freq_range=(90, 3000), idx_step=1, n_harmonics=5):
    idx_start = np.where(freqs < freq_range[0])[-1][-1]
    idx_end = np.where(freqs < freq_range[-1])[-1][-1] 

    freq_of_power = []
    powers = []
    for freq_idx in range(idx_start, idx_end, idx_step):
        power_ = total_pitch_power_adjusted(sxx, freq_idx, bandwidth_idx=1, n_harmonics=n_harmonics)
        powers.append(power_)
        freq_of_power.append(freqs[freq_idx])
    
    return freq_of_power, powers


def total_pitch_power_adjusted(sxx, fundamental_idx, bandwidth_idx=1, n_harmonics=5):
    """
    TODO: index out of range at high frequency
    """
    nth_harmonics = np.arange(1, n_harmonics+1, dtype=int)
    total_power = 0

    approx_half_cycle_size = fundamental_idx//2
    #bandwidth_idx = fundamental_idx//6

    n = 1
    for n in nth_harmonics:
        i = n*fundamental_idx

        if (i+approx_half_cycle_size) >= len(sxx):
            break
        
        cycle_mean = sxx[i-approx_half_cycle_size: i+approx_half_cycle_size+1].mean()
        total_power += sxx[i-bandwidth_idx: i+bandwidth_idx+1].mean() - cycle_mean
    return total_power/(n-1)


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