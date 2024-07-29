import onnxruntime
import scipy as sp

from numpy.lib.stride_tricks import as_strided 
class RunONNX: 
    def __init__(self, file):
        self.ort_session = onnxruntime.InferenceSession(file, providers=['CPUExecutionProvider'])

        self.inputs = self.ort_session.get_inputs()
        self.input_names = [i.name for i in self.inputs]

    def __call__(self, *x):
        return self.ort_session.run(None, dict(zip(self.input_names, x)))



class RunModel:
    def __init__(self, model_file, model_fs=16000, window_size=6400):
        self.model = RunONNX(model_file)
        self.model_fs = model_fs
        self.window_size = window_size
        

    def striding(self, sig, fs, overlap_p=0.25):
        # sig: [ts]
        sig = sp.signal.resample(sig, round(len(sig)/fs*self.model_fs))

        step_size = round((1-overlap_p)*self.window_size)

        return as_strided(sig, shape=((len(sig)//step_size-1), self.window_size), strides=(sig.strides[0]*step_size,sig.strides[0]))
        
    def feed(self, sig, fs, overlap_p=0.25):
        # sig: [ts]
        x = self.striding(sig, fs, overlap_p)

        return self.model(x[None, ...])
