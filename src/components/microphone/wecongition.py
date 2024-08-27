from typing import Callable, List, Union, cast
from typing_extensions import deprecated
import numpy as np

from components.syncronisation import CallChannel, ComponentInterface, component, declare_function_handler, loop, samples_producer, declare_method_handler
from components.logger import LoggerComponent, add_time
from data_collection.data_collection import LoggerSet, Logger
from .viz import get_table, get_power_plot
from .microphone import MicrophoneReader
from plotly.graph_objects import FigureWidget

from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from scipy.fft import fft, fftfreq
import onnxruntime
from .microphone import MicrophoneComponent

import torchaudio as ta
import torch 
tch = torch
tafn = ta.functional

@staticmethod
def tensor(sig: np.ndarray, dtype=tch.float32):
    try:
        return tch.tensor(sig, dtype=dtype)
    except:
        return tch.tensor(sig.copy(), dtype=dtype)
        


class RunONNX: 
    def __init__(self, file):
        self.ort_session = onnxruntime.InferenceSession(file, providers=['CPUExecutionProvider'])

        self.inputs = self.ort_session.get_inputs()
        self.input_names = [i.name for i in self.inputs]

    def __call__(self, *x):
        return self.ort_session.run(None, dict(zip(self.input_names, x)))

@component
class Wecognition(ComponentInterface):
    def __init__(self, model_path, get_signal: CallChannel):
        self.model = RunONNX(model_path)

        self.get_signal = declare_method_handler(get_signal, MicrophoneComponent.get_signal)

    @loop
    @samples_producer(typecodes=['d'], default_values=[0])
    def main_loop(self):
        ok, sig = self.get_signal.call(0)()
        sig = cast(np.ndarray, sig)
        sig = sig.astype(np.float32)
        sig /= (sig**2).sum()**(1/2)

        return self.model(np.array(sig)[None, None, :])


class WecognitionModel:
    def __init__(self, model_path):
        self.model= RunONNX(model_path)

    def __call__(self, sig):
        if not len(sig) == self.siglen:
            print(f'wrong signal length: {len(sig)}')
            return False
        sig = cast(np.ndarray, sig)
        sig = sig.astype(np.float32)
        sig /= (sig**2).sum()**(1/2)

        return self.model(np.array(sig)[None, None, :])[-1] > 0.7
    
    @property
    def fs(self):
        return 16000

    @property
    def siglen(self):
        return int(16000*0.4)


@component
class WeDrive(ComponentInterface):
    def __init__(self, model_path, mic: MicrophoneReader):

        self.mic = mic
        self.weee_model = WecognitionModel(model_path)
        assert self.weee_model.fs == self.mic.sample_rate
        self.fs = self.weee_model.fs

        self.to_get_pitch = False


        self.during_eee = False

        self.prev_pitch_f = 0
        self.max_pitch_f_change = 25


    @loop
    @samples_producer(typecodes=['d', 'd'], default_values=[0, 0])
    def main_loop(self):
        
        # the signal is supposed to be around 1 second long
        # detect pitch frequency gives f0 at the latest frame
        sig = self.mic.get_signal()

        n_sec_before = 0.2
        idx = int(n_sec_before*self.fs)
        if self.during_eee: 
            new_pitch_f = self.detect_pitch_frequency(
                sig, self.fs, 0.5, 600,
            )

            if abs(new_pitch_f - self.prev_pitch_f) < self.max_pitch_f_change:
                self.prev_pitch_f = new_pitch_f
            else:
                self.during_eee = False
                self.prev_pitch_f = 0



        elif self.weee_model(sig[(-self.weee_model.siglen-idx):(len(sig)-idx)]): 
            
            # TODO: let's hope the we does not affect the pitch detection
            # pitch detection has to be done at the same time as the we is detected
            # otherwise a short burst of we would cause a problem 
            # An alternative is to require high power during the weee because the car 
            # runing noise is not expected 

            self.prev_pitch_f = self.detect_pitch_frequency(
                sig, self.mic.sample_rate, 0.2, 600,
            )
            self.during_eee = True

        else: 
            # self.during_eee would be false here
            pass

        
        if self.prev_pitch_f:
            # during eee
            # wrap the pitch frequency to pitch

            pitch = self.prev_pitch_f
            angular_velocity = angle2omega(pitch, 220, 120, 180)

            return 100, angular_velocity
        else:
            # not during eee
            return 0, 0


    
    @classmethod
    def entry(cls, model_path):
        mic = MicrophoneReader(None, approx_frame_duration=0.05, n_frame_buffer=20)

        return cls(model_path, mic)


    @staticmethod
    def detect_pitch_frequency(sig: np.ndarray, fs:int, frame_time:float, freq_high:int):
        sig_t = tensor(sig, dtype=tch.float32)
        pitches = tafn.detect_pitch_frequency(
            sig_t, 
            fs,
            frame_time=frame_time, 
            win_length=3, 
            freq_high=freq_high
        )
        return pitches.numpy()[..., -1]

    

def freq_to_pitch(freq) -> Union[float, np.ndarray]:
    return (np.log2(freq) % 1) * 360 


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
