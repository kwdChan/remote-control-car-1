
from typing import Union, Tuple
from pvrecorder import PvRecorder
from scipy.fft import fft, fftfreq
import numpy as np


class MicrophoneReader:
    def __init__(self, device_index: Union[int, None], frame_length, signal_nframe, sxx_nframe, device_name_match='usb'):
        if device_index is None:
            device_index, device_name = MicrophoneReader.try_get_devices(device_name_match)
            print(f'using device: {device_name}')

        # recorder 
        recorder = PvRecorder(frame_length=frame_length, device_index=device_index)
        recorder.start()
        self.recorder  = recorder 

        # values 
        freqs = fftfreq(recorder.frame_length*signal_nframe, 1/recorder.sample_rate)
        n_freqs = int(len(freqs) //2)
        freqs = freqs[:n_freqs] # type: ignore
        frame_duration = recorder.frame_length / recorder.sample_rate

        # buffer 
        self.signal_buffer = [np.zeros(frame_length) for _ in range(signal_nframe)]
        self.sxx_buffer = [np.zeros(n_freqs) for _ in range(sxx_nframe)]


        # values 
        self.freqs = freqs
        self.n_freqs = n_freqs
        self.frame_length = frame_length
        self.frame_duration = frame_duration


    def sample(self):
        """
        fft overlaps
        """

        x = self.recorder.read()
        
        self.signal_buffer = self.signal_buffer[1:] + [x]

        y = fft(np.concatenate(self.signal_buffer))[:self.n_freqs]
        sxx = np.log(y.real**2 + y.imag**2) # type: ignore 
        
        self.sxx_buffer =  self.sxx_buffer[1:] + [sxx] 

    @staticmethod
    def show_devices():
        for index, device in enumerate(PvRecorder.get_available_devices()):
            print(f"[{index}] {device}")

    @staticmethod
    def try_get_devices(match_str:str) -> Tuple[int, str]:
        devices = PvRecorder.get_available_devices()
        for index, device in enumerate(devices):
            if match_str.lower() in device.lower():
                return index, device
        return 0, devices[0]




