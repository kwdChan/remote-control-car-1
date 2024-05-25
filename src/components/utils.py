from multiprocessing import Value
from typing import List, Dict, Literal, Union, Tuple, Any, TypeVar, cast, Generic
from multiprocessing.connection import Connection
from typing_extensions import deprecated
from data_collection.data_collection import Logger
import time
T = TypeVar('T')



def loop_for_n_sec(loop, callback, n_sec):
    """
    for notebook use
    """    
    start_t = time.monotonic()
    while (time.monotonic() - start_t) < n_sec:
        loop()
    callback()