from contextlib import ContextDecorator
from typing import Callable, List, Any
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


def mon_samples(samples: List[Callable[[], Any]], interval=0.1):
    while True:
        line = ""
        for s in samples:
            line = line + str(s()) + ', '
        print(line[:-2], end='           \r')
        time.sleep(interval)