from contextlib import ContextDecorator
from typing import Callable
import time

class Timer:
    timelapsed: float
    def __init__(self, time_func: Callable[..., float] = time.perf_counter):
        self.time_func = time_func
        
    def __enter__(self):
        self.t0 = self.time_func()
        return self

    def __exit__(self, *exc):
        self.timelapsed = self.time_func()-self.t0
        return False
