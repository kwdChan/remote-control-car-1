from typing import Callable, List, Union, cast
from typing_extensions import deprecated
import numpy as np

from components.syncronisation import ComponentInterface, component, loop, samples_producer
from data_collection.data_collection import LoggerSet, Logger
from .viz import get_table, get_power_plot
from .microphone import MicrophoneReader
from plotly.graph_objects import FigureWidget

from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from scipy.fft import fft, fftfreq


class PitchPersistence:
    """
    this is done because i didn't want the occasional nan to keep stopping the car 
    """
    def __init__(self, min_power=375, max_deg_diff=60, n_frames=3):
        self.min_power = min_power
        self.max_deg_diff = max_deg_diff
        self.n_frames = n_frames

        self.pitch_hist = [np.nan]*n_frames
        self.is_valid_hist = [False]*n_frames
    
    def new_frame(self, pitch, power):

        self.pitch_hist = self.pitch_hist[1:] + [pitch]

        max_angle_diff = self.get_max_angle_diff(self.pitch_hist)

        
        this_frame_valid = (max_angle_diff<self.max_deg_diff) and (power>self.min_power)

        self.is_valid_hist = self.is_valid_hist[1:] + [this_frame_valid]

        return all(self.is_valid_hist), np.nanmedian(self.pitch_hist)
            
    
    @staticmethod
    def get_max_angle_diff(degrees: List|np.ndarray):
        """
        TODO: Terribly inefficient 
        - Only the angle difference of the latest step need to be checked 
        - Can be vectorised
        """
        max_diff = 0
        for i in range(len(degrees)-1):
            for j in range(i, len(degrees)):
                max_diff = np.nanmax([max_diff, angle_deg_between(degrees[i], degrees[j])])
        return max_diff
        

@component
class PitchAngularVelocityController(ComponentInterface):
    def __init__(self, microphone: MicrophoneReader, speed):


        expected_sig_length = microphone.frame_length*microphone.n_frame_buffer
        pitch_detector = PitchDetector(expected_sig_length, microphone.sample_rate)


        # values
        self.pitch_detector = pitch_detector
        self.microphone = microphone
        self.speed = speed

        # state 
        self.pitch_persistence_check = PitchPersistence()

    @loop
    @samples_producer(typecodes=['d', 'd'], default_values = [0, 0])
    def step(self):

        sig = self.microphone.get_signal()
        if not len(sig)==self.pitch_detector.sig_len:
            return 0, 0
        

        pitch, power = self.pitch_detector.sig2pitch(sig)

        valid, pitch = self.pitch_persistence_check.new_frame(pitch, power)

        # this is done because pitch can be nan 
        if valid:
            speed = self.speed
            angular_velocity = self.angle2omega(pitch, 220, 120, 180)
        else: 
            speed = 0
            angular_velocity = 0 
        


        # self.logger.log_time("AhhhhhWheelController")
        # self.logger.log('speed_or_invalid', speed)
        # self.logger.log('median_pitch', np.nanmedian(self.pitch_hist))
        # self.logger.log('balance', balance)

        return cast(float, speed), cast(float, angular_velocity)


    @staticmethod
    def angle2omega(angle_deg, centre, angle_scale_oneside, omega_scale_oneside):
        """
        angle_scale_oneside: (0, 180]
        angle_centred: (-180, 180]


        mapped an angle to a number 
            when (angle_deg == centre + angle_scale_oneside) -> omega_scale_oneside
            when (angle_deg == centre + 0)                   -> 0
            when (angle_deg == centre - angle_scale_oneside) -> -omega_scale_oneside
        """
        angle_centred = (angle_deg - centre) % 360
        if angle_centred > 180:
            angle_centred = angle_centred - 360

        return angle_centred/(angle_scale_oneside/omega_scale_oneside)

    @classmethod
    def entry(cls, speed=100, **kwargs):
        """
        approx_frame_duration: 50ms
        signal_nframe: 5
        """
        # 50ms
        mic = MicrophoneReader(None, approx_frame_duration=0.05, n_frame_buffer=5)

        return cls(mic, speed, **kwargs)




class PitchDetector:
    def __init__(self, sig_len, sample_rate):
        freqs = fftfreq(sig_len, 1/sample_rate)
        n_freqs = int(len(freqs) //2)
        freqs = freqs[:n_freqs]

        self.freqs = freqs
        self.n_freqs = n_freqs
        self.sig_len = sig_len
    
    def sig2pitch(self, sig, freq_range=(90, 3000) ):
        assert len(sig)==self.sig_len

        y = fft(sig)[:self.n_freqs]
        sxx = np.log(y.real**2 + y.imag**2) # type: ignore 

        basefreq, pitch_power = find_pitch_with_fft(sxx,  self.freqs, freq_range=freq_range)

        pitch = freq_to_pitch(basefreq) 

        return pitch, pitch_power



def angle_deg_between(a1, a2):
    return min((a1-a2)%360, (a2-a1)%360)


def rolling_mean(arr, kernel=np.ones(3)/3): 
    return np.correlate(arr, kernel, mode='same')
    

## TODO: i need to change to pitch detection algo

def adjust(sxx, bandwidth=45):
    return sxx - rolling_mean(sxx, np.ones(bandwidth)/bandwidth) 

@deprecated('i need to change the pitch detection algo')
def fft_power_spectra(sxx, freqs):
    sxx = sxx-sxx.mean()
    y = fft(sxx)
    freqs = fftfreq(len(freqs), freqs[1]-freqs[0])

    pitch_power = y.real**2+y.imag**2 # type: ignore
    pitch_power = cast(np.ndarray, pitch_power)
    return 1/freqs[:len(freqs)//2], pitch_power[:len(freqs)//2]

@deprecated('i need to change the pitch detection algo')
def fft_power_spectra_masked_by_adjusted(sxx, freqs):
    f, adjusted_sxx = fft_power_spectra(rolling_mean(adjust(sxx, bandwidth=20)), freqs)
    f, sxx  = fft_power_spectra(sxx, freqs)

    # TODO: constant
    return f, sxx*(adjusted_sxx>(20e3)) 

@deprecated('i need to change the pitch detection algo')
def find_pitch_with_fft(sxx, freqs, freq_range=(90, 600)):
    freq_pitch, sxx_pitch = fft_power_spectra_masked_by_adjusted(sxx, freqs)
    bidx = (freq_pitch>freq_range[0]) & (freq_pitch<freq_range[1])
    argmax = np.argmax(sxx_pitch[bidx])

    return freq_pitch[bidx][argmax], sxx_pitch[bidx][argmax]


def freq_to_pitch(freq) -> Union[float, np.ndarray]:
    return (np.log2(freq) % 1) * 360 
