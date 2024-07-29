from __future__ import annotations
from typing import Optional, Union, Tuple, List, Callable
from pvrecorder import PvRecorder
from scipy.fft import fft, fftfreq
import numpy as np
from scipy import signal as ss


from components.syncronisation import ComponentInterface, create_thread, component, loop, rpc, samples_producer, sampler


def get_microphone_reader(device_index: Union[int, None], frame_length, device_name_match='usb'):
    if device_index is None:
        device_index, device_name = MicrophoneReader.try_get_devices(device_name_match)
        print(f'using device: {device_name}')

@component
class MicrophoneComponent(ComponentInterface):
    def __init__(self, mics: JointMicrophoneReader, target_frame_duration:float, new_fs:int):
        """
        output a frame of signal at a different sampling rate to the microphone 
        """
 
        self.mics = mics
        self.fs = new_fs

        self.required_n_sample = np.ceil(target_frame_duration*mics.sample_rate)
        self.n_frame_required = np.ceil(self.required_n_sample/mics.frame_length)

        self.actual_frame_length = mics.frame_length*self.n_frame_required
        self.actual_frame_duration = self.actual_frame_length/new_fs

        self.resample_ratio = new_fs/mics.sample_rate


        # states
        self.channels = [[[0]*mics.frame_length]*self.n_frame_required]*len(mics)

        # 
        self.mics.start(self.new_frame)

    def new_frame(self, idx:int, sig: List[float]):
        self.channels[idx].remove(0)
        self.channels[idx].append(sig)

    @rpc()
    def get_raw_signal(self, ch):
        return np.concatenate(self.channels[ch])

    def resample(self, x):
        return ss.resample(x, round(len(x)*self.resample_ratio))
    @rpc()
    def get_signal(self, ch):
        return self.resample(self.get_raw_signal(ch))

    @rpc()
    def get_sig_diff(self, ch0, ch1):
        return self.resample(self.get_raw_signal(ch0) - self.get_raw_signal(ch1))

    @rpc()
    def get_sample_rate(self):
        return self.mics.sample_rate

    @classmethod
    def entry(cls, device_indices):
        mics = JointMicrophoneReader(device_indices, frame_length=512)
        return cls(mics, target_frame_duration=0.4, new_fs=16000) 
        
    
from threading import Barrier
class JointMicrophoneReader:
    def __init__(self, device_indices: List[int], frame_length=512):
        # TODO: assert that they all have the same sampling rate 
        # TODO: allow frame dropping in one microphone skips one frame for some reasons
        assert len(device_indices)
        self.recorders: List[PvRecorder] = [
            PvRecorder(frame_length=frame_length, device_index=i)
            for i in device_indices
        ]

        self.frame_length = frame_length
        
        self.sample_rate = self.recorders[0].sample_rate
        self.barrier = Barrier(len(self.recorders))
        self.to_sample = True

    def __len__(self):
        return len(self.recorders)


    def __get_frame(self, recorder_index, callback):
        self.barrier.wait()
        sig = self.recorders[recorder_index].read()
        callback(recorder_index, sig)
    
    def start(self, callback: Callable[[int, List[float]], None]):
        
        def get_frame_continously(idx):

            while self.to_sample: 
                self.__get_frame(idx, callback)


        for r in self.recorders: r.start()
        self.ts = [create_thread(get_frame_continously, i) for i in range(len(self.recorders))]

        for t  in self.ts: t.start()

    def stop(self): 
        self.to_sample = False


    


    




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




