
from typing import Optional, Union, Tuple
from pvrecorder import PvRecorder
from scipy.fft import fft, fftfreq
import numpy as np

from components.syncronisation import create_thread

def get_microphone_reader(device_index: Union[int, None], frame_length, device_name_match='usb'):
    if device_index is None:
        device_index, device_name = MicrophoneReader.try_get_devices(device_name_match)
        print(f'using device: {device_name}')

    

class MicrophoneReader:
    """
    connect to the microphone, return the signal

    """
    def __init__(self, device_index: Union[int, None], approx_frame_duration:Optional[float]=None, frame_length:Optional[int]=None, device_name_match='usb', n_frame_buffer=5):
        if device_index is None:
            device_index, device_name = MicrophoneReader.try_get_devices(device_name_match)
            print(f'using device: {device_name}')

        assert not (approx_frame_duration and frame_length), "only one of these two should be provided"
        if frame_length is None:
            assert approx_frame_duration
            guess_sample_rate = self.get_sample_rate(device_index)
            frame_length = int(guess_sample_rate*approx_frame_duration)

        # recorder 
        recorder = PvRecorder(frame_length=frame_length, device_index=device_index)
        recorder.start()

        frame_duration = recorder.frame_length / recorder.sample_rate

        # values 
        self.recorder  = recorder 

        # for public access only
        self.frame_length = frame_length
        self.frame_duration = frame_duration
        self.n_frame_buffer = n_frame_buffer
        self.sample_rate = recorder.sample_rate

        # states
        self.__signal_buffer = [[]]*n_frame_buffer
        self.__to_sample: bool

        # start 
        self.start()


    def start(self):
        self.__to_sample = True
        self.sampling_thread = create_thread(self.__sample)
        self.sampling_thread.start()

    def __sample(self):
        while self.__to_sample: 
            x = self.recorder.read()
            self.__signal_buffer = self.__signal_buffer[1:] + [x]

    def stop(self):
        # flag the thread to end 
        self.__to_sample = False


    def release(self):
        self.stop()
        self.recorder.delete()

    
    def get_signal(self) -> np.ndarray:
        return np.concatenate(self.__signal_buffer)
    

    @staticmethod
    def get_sample_rate(device_index, _sham_frame_length=1000)->int:
        recorder = PvRecorder(frame_length=_sham_frame_length, device_index=device_index)
        try: 
            sample_rate = recorder.sample_rate
        finally:
            recorder.delete()
        return sample_rate


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




