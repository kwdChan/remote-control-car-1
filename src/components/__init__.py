from abc import abstractmethod
from enum import Enum
import time
from typing import Callable, Tuple, TypeVar, cast, get_origin, get_args, Union, Any, Optional, Dict, List
import threading

from multiprocessing import Value, Process

from multiprocessing import Manager, Value, Array
from multiprocessing.managers import BaseProxy, BaseManager, SyncManager

from threading import Lock as ThreahingLock



from multiprocessing import Semaphore, Lock
from multiprocessing.synchronize import Lock as LockType, Semaphore as SemaphoreType
from multiprocessing.managers import BaseProxy, BaseManager, SyncManager, ListProxy, DictProxy


from typing_extensions import deprecated
import warnings
import inspect
import numpy as np
from data_collection.data_collection import Logger
from typing import Type
import array



T = TypeVar('T')
def get_switch(v1: Callable[[], T], v2: Callable[[], T], use1: Callable[[], bool]) -> Callable[[], T]:

    return lambda: v1() if use1() else v2()


def get_switches(v1: List[Callable[[], T]], v2: List[Callable[[], T]], use1: Callable[[], bool]) -> List[Callable[[], T]]:
    
    values = []
    for _v1, _v2 in zip(v1, v2):
        values.append(
            get_switch(_v1, _v2, use1 )
        )
    return values

def shared_list(typecodes, default_values):
    """TODO: testing needed"""
    v_list = [Value(tc, dv) for tc, dv in zip(typecodes, default_values)]

    def reader():
        return [v.value for v in v_list]

    def assigner(slicer, value):
        v_list[slicer] = value

    return reader, assigner


def shared_value_v2(typecode: Any, default_value:Any=0) -> Tuple[Callable, Callable]:

    v = Value(typecode, default_value)
    def reader():
        return v.value

    def assigner(value):
        v.value=value

    return reader, assigner


def shared_np_array_v2(typecode: Any, default_value: np.ndarray) -> Tuple[Callable, Callable]:

    flatten_arr = Array(typecode, default_value.ravel().tolist())
    dim = default_value.shape

    def assigner(value: np.ndarray):
        flatten_arr[:] = array.array(typecode, value.ravel()) # type: ignore

    def reader():
        return np.array(flatten_arr[:]).reshape(dim)

    return reader, assigner


class CallChannel:
    
    @staticmethod
    def default_kwargs_reader(kwargs:Dict):
        return kwargs

    @staticmethod
    def default_kwargs_writer(store):
        return store

    def __init__(
        self, 
        name, 
        manager: SyncManager, 
        kwargs_reader: Optional[Callable[[Dict], Dict]]=None, 
        kwargs_writer: Optional[Callable[[Dict], Dict]]=None, 
    ):

        self.name = name
        self.manager = manager

        self.kwargs_reader = kwargs_reader if kwargs_reader else CallChannel.default_kwargs_reader 
        self.kwargs_writer = kwargs_writer if kwargs_writer else CallChannel.default_kwargs_writer

        # states 
        self.call_pending: ListProxy[Tuple[int, Any]] = manager.list()
        self.res_pending: DictProxy[int, Any] = manager.dict()

        self.response_locks: DictProxy[int, Any] = manager.dict()

        self.call_id = -1
        self.call_semaphore = Semaphore(0)

    def call_no_return(self, **kwargs):
        self.call_id += 1 
        self.call_pending.append((self.call_id, self.kwargs_writer(kwargs)))
        self.call_semaphore.release()
        

    def call(self, **kwargs):
        self.call_id += 1 
        self.call_pending.append((self.call_id, self.kwargs_writer(kwargs)))

        lock = self.manager.Lock()
        lock.acquire()
        self.response_locks[self.call_id] = lock
        
        # CANNOT RELEASE THE LOCK BEFORE THE self.response_locks[self.call_id] IS SET
        self.call_semaphore.release()

        this_call_id = self.call_id
        def await_result():

            lock.acquire()
            self.response_locks.pop(this_call_id)

            return self.kwargs_reader(self.res_pending.pop(this_call_id))

        return await_result

    def await_and_handle_call(self, handler: Callable[[], Any]):
        self.call_semaphore.acquire()
        call_id, kwargs = self.call_pending.pop(0)
        
        res = handler(**kwargs)

        l = self.response_locks.get(call_id)
        if l is not None:
            self.res_pending[call_id] = self.kwargs_writer(res)
            l.release()



    def message_handling_loop(self, handler: Callable[[], Any]):
        while True:
            self.await_and_handle_call(handler)

    def start_message_handling_thread(self, handler: Callable[[Any], Any]):
        # TODO: saving t to self.t may make the object unpickleable?
        # TODO: have a pool of threads to consume the messages as fast as possible 
        t = threading.Thread(target=self.message_handling_loop, kwargs=dict(handler=handler))
        t.start()
        return t

#

def loop_func(func, ideal_interval):
    while True:
        st = time.monotonic()
        func()
        et = time.monotonic()
        
        itv_to_sleep = ideal_interval-(et-st)
        if itv_to_sleep <= 0:
            warnings.warn("the real interval is >= ideal_interval")
        time.sleep(max(0, itv_to_sleep))


class ComponentInterface:
    rpc_list = []
    samplers = []
    sample_producers = []

ComponentSubtype = TypeVar('ComponentSubtype', bound=ComponentInterface)

def component():
    """
    """
    
    def dec(cls: Type[ComponentSubtype]):    

        cls.rpc_list = []
        
        cls.samplers = []
        cls.sample_producers = []

        for k, v in cls.__dict__.items():
            if not hasattr(v, "handler_type"): continue

            
            if v.handler_type == 'rpc':
                cls.rpc_list.append(v)

            elif v.handler_type == 'sample-related':
                if 'sampler' in v.roles:
                    cls.samplers.append(v)

                if 'sample-producer' in v.roles:
                    cls.sample_producers.append(v)
            else:
                raise NotImplementedError

            if len(cls.sample_producers) >1:
                raise NotImplementedError

        return cls
    return dec


def rpc(
    kwargs_reader: Optional[Callable[[Any], Any]] = None, 
    kwargs_writer: Optional[Callable[[Any], Any]] = None
) -> Callable[[Callable], Callable]:
    
    def dec(func: Callable[[Any, Any], Any]):
        """
        the CallChannel is instantiated by this component and passed on to the other components
        """
        if not hasattr(func, "roles"):
            setattr(func, "roles", [])        

        getattr(func, "roles").append("rpc")

        
        setattr(func, "kwargs_reader", kwargs_reader)
        setattr(func, "kwargs_writer", kwargs_writer)

        return func 

    return dec



SampleReader = Callable[[], Any]
SampleWriter = Callable[[Any], None]

SampleSetupFunction = Callable[[List[Any], List[Any]], Tuple[List[SampleReader], List[SampleWriter]]]

def numpy_sample_setup(typecodes, default_values): 
    readers, writers = [], []
    for tc, dv in zip(typecodes, default_values):
        if isinstance(dv, np.ndarray): 
            r, w = shared_np_array_v2(tc, dv)
        else:
            r, w = shared_value_v2(tc, dv)
        readers.append(r)
        writers.append(w)

    return readers, writers

# TODO: this requires the shape of the array to be pre-defined in the decorator, which is not ideal 
def samples_producer(typecodes:List[Any], default_values:List[Any], setup_function:SampleSetupFunction=numpy_sample_setup):

    def setup(): 
        return setup_function(typecodes, default_values)
        
    def dec(func: Callable): 

        setattr(func, "setup_func", setup)

        if not hasattr(func, "roles"):
            setattr(func, "roles", [])
        getattr(func, "roles").append("sample-producer")
        
        return func

    return dec


def sampler(func: Callable):
    """
    TODO: 
    """

    if not hasattr(func, "roles"):
        setattr(func, "roles", [])
    getattr(func, "roles").append("sampler")
    
    return func


def sampler_wrapper(func: Callable, args: List[SampleReader]=[], kwargs: Dict[str, SampleReader]={}):
    def wrapped_func():
        args_realised = [f() for f in args]
        kwargs_realised = {k:f() for k, f in kwargs.items()}
        return func(*args_realised, **kwargs_realised)
    return wrapped_func

def sample_producer_wrapper(func: Callable, sampler_writers: List[SampleWriter]):
    def wrapped_func(*args, **kwargs):
        return_values = func(*args, **kwargs)

        if not len(sampler_writers):
            return_values = () 

        if len(sampler_writers)==1:
            return_values = (return_values,)
        
        for w, v in zip(sampler_writers, return_values):
            w(v)
    
    return wrapped_func


def loop(func: Callable):
    if not hasattr(func, "roles"):
        setattr(func, "roles", [])
    getattr(func, "roles").append("loop")
        

def target(
    component_class: Type[ComponentInterface], 
    instantiater: Callable[..., ComponentInterface], 

    incoming_sample_readers: List[Callable[[], Any]], 
    incoming_rpcs:Dict[str, CallChannel], 

    outgoing_sample_writers: List[Callable[[Any], None]],
    outgoing_rpcs:Dict[str, CallChannel], 

    loop: Callable[[Callable], None], 
    loop_kwargs, 
    init_kwargs, 
    ):

    obj = instantiater(**init_kwargs, **outgoing_rpcs)

    ## all the wrapping 
    # wrap the sampler 
    for f in component_class.samplers:
        method = getattr(obj, f.__name__)
        setattr(obj, f.__name__, sampler_wrapper(method, incoming_sample_readers))

    # wrap the sample producer
    for f in component_class.sample_producers:
        method = getattr(obj, f.__name__)
        setattr(obj, f.__name__, sample_producer_wrapper(method, outgoing_sample_writers))


    ## all the triggers

    # the rpc (set up after the wrapping)
    for name, chan in incoming_rpcs.items():
        chan.start_message_handling_thread(getattr(obj, name))

    # TODO: set interval trigger 

    
    


def create_component_starter(
    component_class:Type[ComponentInterface], 
    manager: SyncManager, 
    loop, init_kwargs, loop_kwargs, instantiater=None
    ):
    """
    1. create a message channel for each method with @rpc, identified by function name
    2. create a event broadcaster for each event named by @component TODO: may create a @event_producer decorator
    3. create a list of readers and writers of the samples using the setup function attached by @sample_producer


    Expect:
    1. message channel for rpc of the other component, matached to the name of the variable in the instantiator 
    2. a list of sample readers, matched by argument position of the method with @sampler
    3. 

    """



    assert hasattr(component_class, "rpc_list")
    assert hasattr(component_class, "event_handlers")
    assert hasattr(component_class, "events_to_produce")
    
    incoming_rpcs: Dict[str, CallChannel] = {}


    # create a message channel for each method with @rpc, identified by function name
    for f in component_class.rpc_list:
        
        chan = CallChannel(f.__qualname__, manager, f.kwargs_reader, f.kwargs_writer) 
        incoming_rpcs[f.__name__] = chan

        
    #component_class.samplers
    
    if len(component_class.sample_producers) > 1:
        raise NotImplementedError

    if len(component_class.sample_producers): 
        sample_producer = component_class.sample_producers[0]

        assert hasattr(sample_producer, "setup_func")
        outgoing_value_samplers, outgoing_value_assigners = sample_producer.setup_func()

    else: 
        outgoing_value_samplers, outgoing_value_assigners = [], []


    # this feels so wrong haha
    class ComponentStarter:
        def __init__(self, outgoing_value_samplers, incoming_rpc) :
            self.process: Process


            self.incoming_rpc: Dict[str, CallChannel] = incoming_rpc
            self.outgoing_samples:List[SampleReader] = outgoing_value_samplers


            self.outgoing_rpc: Dict[str, CallChannel] = {}
            self.incoming_samples:List[SampleReader] = []

        def start(self):


            self.process = Process(
            target=target, 
            kwargs=dict(
                component_class = component_class, 
                instantiater = instantiater if instantiater else component_class, 
                incoming_value_samplers=self.incoming_samples, 

                incoming_rpcs=incoming_rpcs, 
                outgoing_value_assigners=outgoing_value_assigners, 
                outgoing_rpcs=self.outgoing_rpc, 

                loop=loop, 
                loop_kwargs=loop_kwargs, 
                init_kwargs=init_kwargs,
                )
            )
            self.process.start()


        def register_outgoing_rpc(self, outgoing_rpc):
            self.outgoing_rpc = outgoing_rpc


        def register_incoming_samples(self, incoming_samples: List[SampleReader]):
            self.incoming_samples = incoming_samples 

            
    starter = ComponentStarter(outgoing_value_samplers, incoming_rpcs)

    return starter



@component()
class MyTestComponent(ComponentInterface):
    def __init__(self):
        self.idx = -1

    @samples_producer(['d', 'd'], [0, np.zeros((4,4))])
    @sampler
    def step(self, idx_other, arr_other):
        self.idx += 1

        
        return self.idx, np.random.random((4,4)) # type: ignore
        



    @rpc()
    def com1_rpc(self, msg):

        print(f'com1 received msg: {msg}', flush=True)

        return msg


def testing_function():
    m = Manager()

    sample_reader, sample_writer = numpy_sample_setup(['d', 'd'], [0, np.zeros((4,4))])
    
    starter1 = create_component_starter(

        MyTestComponent, manager=m, loop=loop, init_kwargs={}, loop_kwargs={'ideal_interval': 1}, instantiater=None
    )

    starter1.register_incoming_samples(sample_reader)
    starter1.register_outgoing_rpc({})
    
    starter1.start()

    return starter1, sample_writer

