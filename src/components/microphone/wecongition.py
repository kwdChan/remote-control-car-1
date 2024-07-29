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

class RunONNX: 
    def __init__(self, file):
        self.ort_session = onnxruntime.InferenceSession(file, providers=['CPUExecutionProvider'])

        self.inputs = self.ort_session.get_inputs()
        self.input_names = [i.name for i in self.inputs]

    def __call__(self, *x):
        return self.ort_session.run(None, dict(zip(self.input_names, x)))


class Wecognition:
    def __init__(self, model_path, get_signal: CallChannel):
        self.model = RunONNX(model_path)

        self.get_signal = declare_method_handler(get_signal, MicrophoneComponent.get_signal)

    @loop
    @samples_producer(typecodes=['d'], default_values=[0])
    def main_loop(self):
        ok, sig = self.get_signal.call(0)()

        return self.model(np.array(sig)[None, None, :])

        