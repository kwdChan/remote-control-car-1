
import time
from typing import Callable, Concatenate, Generic, Literal, Tuple, TypeVar, TypedDict, cast, get_origin, get_args, Union, Any, Optional, Dict, List
import threading

from multiprocessing import Value, Process

from multiprocessing import Manager, Value, Array
from multiprocessing.managers import BaseProxy, BaseManager, SyncManager

from threading import Lock as ThreahingLock



from multiprocessing import Semaphore, Lock, Condition
from multiprocessing.synchronize import Lock as LockType, Semaphore as SemaphoreType
from multiprocessing.managers import BaseProxy, BaseManager, SyncManager, ListProxy, DictProxy


from typing_extensions import deprecated
import warnings
import inspect
import numpy as np
from data_collection.data_collection import Logger
from typing import Type, ParamSpec
import array


P = ParamSpec('P')
T = TypeVar('T')
F = TypeVar('F', bound=Callable)

def create_thread(_target: Callable[P, T], /,*args: P.args, **kwargs: P.kwargs):
    """
    start a thread with type check
    """
    return threading.Thread(target=_target, args=args, kwargs=kwargs)

def create_process(_target: Callable[P, T], /,*args: P.args, **kwargs: P.kwargs):
    """
    start a process with type check
    """
    return Process(target=_target, args=args, kwargs=kwargs)

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
        return np.array(flatten_arr[:], dtype = default_value.dtype).reshape(dim)

    return reader, assigner



class CallChannelV2(Generic[P, T]):
    """
    #TODO: testing
    #TODO: V3 with shared memory manager 
    """

    @staticmethod
    def default_args_reader(args:Tuple, kwargs:Dict):
        return args, kwargs

    @staticmethod
    def default_args_writer(args:Tuple, kwargs:Dict):
        return args, kwargs

    def __init__(
        self, 
        name, 
        manager: SyncManager, 
        args_reader: Optional[Callable[[Tuple, Dict], Tuple[Tuple, Dict]]]=None, 
        args_writer: Optional[Callable[[Tuple, Dict], Tuple[Tuple, Dict]]]=None, 
    ):

        self.name = name
        self.manager = manager

        self.args_reader = args_reader if args_reader else self.default_args_reader 
        self.args_writer = args_writer if args_writer else self.default_args_writer

        # states
        ## [(return_idx, params)]
        self.call_pending: ListProxy[Tuple[Optional[int], Any]] = manager.list()

        ## return_idx -> data
        self.res_pending: DictProxy[int, Any] = manager.dict()
        
        self.response_condition = Condition()

        self.call_queue_lock = Lock()
        self.return_addr_lock = Lock()

        # TODO: reuse the address otherwise the channel will dies after the value overflow 
        self.return_addr = Value('L', 1)  # the starting value is 1 so that I don't accidentally do something like `if not return_addr: return None `
        self.call_semaphore = Semaphore(0)


    def call_no_return(self, *args: P.args, **kwargs: P.kwargs)->None:
        with self.call_queue_lock: 
            self.call_pending.append((None, self.args_writer(args, kwargs)))
            self.call_semaphore.release()
        
    #def call(self, *args: P.args, **kwargs: P.kwargs) -> Callable[[], Tuple[Literal[False], None]|Tuple[Literal[True], T]]:
    def call(self, *args: P.args, **kwargs: P.kwargs) -> Callable[[], Tuple[bool, Optional[T]]]:
        """
        TODO: this function must not be used if the return will not be assessed
        the locks and return values will sit there indefinately and accumulate

        """
        with self.return_addr_lock: 
            self.return_addr.value += 1 
            this_return_addr = self.return_addr.value 

        with self.call_queue_lock: 
            self.call_pending.append((self.return_addr.value, self.args_writer(args, kwargs)))
            self.call_semaphore.release()

        def await_result(timeout:Optional[float]=None):
            
            with self.response_condition:
                ready = self.response_condition.wait_for(lambda: (this_return_addr in self.res_pending), timeout=timeout)

            result = self.res_pending.pop(this_return_addr) if ready else None
            return ready, result

        return await_result

    def await_and_handle_call(self, handler: Callable[P, T]):
        self.call_semaphore.acquire()

        with self.call_queue_lock: 
            return_addr, (args, kwargs) = self.call_pending.pop(0)
        
        res = handler(*args, **kwargs) # type: ignore

        if return_addr is None: return

        with self.response_condition: 
            self.res_pending[return_addr] = res
            self.response_condition.notify_all()


    def message_handling_loop(self, handler: Callable[P, T]):
        while True:
            self.await_and_handle_call(handler)

    def start_message_handling_thread(self, handler: Callable[P, T]):
        # TODO: have a pool of threads to consume the messages as fast as possible 
        t = create_thread(self.message_handling_loop, handler=handler)
        t.start()
        return t

@deprecated('it is very slow to create a lock by the manager. use V2 with condition instead ')
class CallChannel(Generic[P, T]):
    

    @staticmethod
    def default_args_reader(args:Tuple, kwargs:Dict):
        return args, kwargs

    @staticmethod
    def default_args_writer(args:Tuple, kwargs:Dict):
        return args, kwargs

    def __init__(
        self, 
        name, 
        manager: SyncManager, 
        args_reader: Optional[Callable[[Tuple, Dict], Tuple[Tuple, Dict]]]=None, 
        args_writer: Optional[Callable[[Tuple, Dict], Tuple[Tuple, Dict]]]=None, 
    ):

        self.name = name
        self.manager = manager

        self.args_reader = args_reader if args_reader else CallChannel.default_args_reader 
        self.args_writer = args_writer if args_writer else CallChannel.default_args_writer

        # states 
        self.call_pending: ListProxy[Tuple[int, Any]] = manager.list()
        self.res_pending: DictProxy[int, Any] = manager.dict()

        self.response_locks: DictProxy[int, Any] = manager.dict()

        self.call_id = Value('L', -1)
        self.call_semaphore = Semaphore(0)
        self.call_lock = Lock()

    def call_no_return(self, *args: P.args, **kwargs: P.kwargs)->None:
        
        with self.call_lock: 
            self.call_id.value += 1 
            self.call_pending.append((self.call_id.value, self.args_writer(args, kwargs)))
            self.call_semaphore.release()
        
    def call(self, *args: P.args, **kwargs: P.kwargs) -> Callable[[], T]:
        """
        TODO: this function must not be used if the return will not be assessed
        the locks and return values will sit there indefinately and accumulate

        """
        with self.call_lock: 
            self.call_id.value += 1 
            self.call_pending.append((self.call_id.value, self.args_writer(args, kwargs)))

            lock = self.manager.Lock()
            lock.acquire()
            self.response_locks[self.call_id.value] = lock
            
            # CANNOT RELEASE BEFORE THE self.response_locks[self.call_id] IS SET
            self.call_semaphore.release()

            this_call_id = self.call_id.value

        def await_result():

            lock.acquire()
            self.response_locks.pop(this_call_id)

            return self.res_pending.pop(this_call_id) 

        return await_result

    def await_and_handle_call(self, handler: Callable[P, T]):
        self.call_semaphore.acquire()

        # TODO: unclear if it's safe to do so. but given there's the GIL, it's probably okay? 
        call_id, (args, kwargs) = self.call_pending.pop(0)
        
        res = handler(*args, **kwargs) # type: ignore

        l = self.response_locks.get(call_id)
        if l is not None:
            self.res_pending[call_id] = res
            l.release()

    def message_handling_loop(self, handler: Callable[P, T]):
        while True:
            self.await_and_handle_call(handler)

    def start_message_handling_thread(self, handler: Callable[P, T]):
        # TODO: have a pool of threads to consume the messages as fast as possible 
        t = create_thread(self.message_handling_loop, handler=handler)
        t.start()
        return t

def declare_function_handler(obj:CallChannel, func: Callable[P, T]) ->  CallChannel[P, T]:
    """
    for type check purposes only
    this function won't work if it is defined within the CallChannel class.
    """
    return obj

def declare_method_handler(obj:CallChannel, method: Callable[Concatenate[Any, P], T]) ->  CallChannel[P, T]:
    """
    ignore the first argment of the callable

    for type check purposes only
    this function won't work if it is defined within the CallChannel class.
    """
    return obj

def declare_function_handler_v2(obj:CallChannelV2, func: Callable[P, T]) ->  CallChannelV2[P, T]:
    """
    for type check purposes only
    this function won't work if it is defined within the CallChannel class.
    """
    return obj

def declare_method_handler_v2(obj:CallChannelV2, method: Callable[Concatenate[Any, P], T]) ->  CallChannelV2[P, T]:
    """
    ignore the first argment of the callable

    for type check purposes only
    this function won't work if it is defined within the CallChannel class.
    """
    return obj

def loop_func(func:Callable, ideal_interval: float):
    while True:
        st = time.monotonic()
        func()
        et = time.monotonic()
        
        itv_to_sleep = ideal_interval-(et-st)
        if itv_to_sleep <= 0:
            warnings.warn("the real interval is >= ideal_interval")
        time.sleep(max(0, itv_to_sleep))



class HandlerRoles(TypedDict):
    rpc: list[Callable]
    sampler: list[Callable]
    sample_producer: list[Callable]
    loop: list[Callable]



class ComponentInterface:
    handler_roles: HandlerRoles = HandlerRoles(rpc=[], sampler=[], sample_producer=[], loop=[])


ComponentSubtype = TypeVar('ComponentSubtype', bound=ComponentInterface)

def component(cls: Type[ComponentSubtype]):    
    """
    decorators

    THESE METHODS ARE TAGED TO BE WARPPED AND THE FUNCTION SIGNATURES MAY CHANGE
    

    """
    possible_roles = list(cls.handler_roles.keys()).copy()
    cls.handler_roles = HandlerRoles(rpc=[], sampler=[], sample_producer=[], loop=[])

    # iterate through each method with "roles" 
    for func in cls.__dict__.values():
        if not hasattr(func, "roles"): continue

        for role in possible_roles: 
            if role in func.roles:
                cls.handler_roles[role].append(func)

    return cls


def rpc(
    args_reader: Optional[Callable[[Any], Any]] = None, 
    args_writer: Optional[Callable[[Any], Any]] = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    
    def dec(func: Callable[P, T]):
        """
        the CallChannel is instantiated by this component and passed on to the other components
        """
        if not hasattr(func, "roles"):
            setattr(func, "roles", [])        

        getattr(func, "roles").append("rpc")

        
        setattr(func, "args_reader", args_reader)
        setattr(func, "args_writer", args_writer)

        return func 

    return dec



SampleReader = Callable[[], Any]
SampleWriter = Callable[[Any], None]
SampleSetupFunction = Callable[Concatenate[List[Any], List[Any], P], Tuple[List[SampleReader], List[SampleWriter]]]

def numpy_sample_setup(typecodes, default_values): 
    readers, writers = [], []

    assert isinstance(default_values, list) or isinstance(default_values, tuple) 
    for tc, dv in zip(typecodes, default_values):
        if isinstance(dv, np.ndarray): 
            r, w = shared_np_array_v2(tc, dv)
        else:
            r, w = shared_value_v2(tc, dv)
        readers.append(r)
        writers.append(w)

    return readers, writers



def samples_producer(setup_function:Callable[..., Tuple[List[SampleReader], List[SampleWriter]]]=numpy_sample_setup, **partial_kwargs):
    """
    defer parameter specification 
    """
    def setup(**kwargs): 
        kwargs = dict(**partial_kwargs, **kwargs)
        return setup_function(**kwargs)
        
    def dec(func: Callable[P, Tuple]) -> Callable[P, Tuple]: 

        setattr(func, "setup_func", setup)

        if not hasattr(func, "roles"):
            setattr(func, "roles", [])
        getattr(func, "roles").append("sample_producer")
        
        return func

    return dec


def sampler(func: Callable[P, T]) -> Callable[P, T]:
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

def sample_producer_wrapper(func: Callable[..., Tuple], sampler_writers: List[SampleWriter]=[]):
    def wrapped_func(*args, **kwargs):
        return_values = func(*args, **kwargs)

        for w, v in zip(sampler_writers, return_values):
            w(v)
    
    return wrapped_func


def loop(func: Callable[P, T])->Callable[P, T]:
    if not hasattr(func, "roles"):
        setattr(func, "roles", [])
    getattr(func, "roles").append("loop")
    return func 
        

def target(
    component_class: Type[ComponentInterface], 
    instantiater: Callable[..., ComponentInterface], 

    incoming_sample_readers: List[Callable[[], Any]], 
    incoming_rpcs:Dict[str, CallChannel], 

    outgoing_sample_writers: List[Callable[[Any], None]],
    outgoing_rpcs:Dict[str, CallChannel], 

    loop_intervals:Dict[str, float], 
    init_kwargs:Dict, 
    ):
    """
    TODO: packages them into smaller reusable functions 
    """

    obj = instantiater(**outgoing_rpcs, **init_kwargs)

    ## all the wrapping 
    # wrap the sampler 
    for f in component_class.handler_roles['sampler']:
        method = getattr(obj, f.__name__)
        setattr(obj, f.__name__, sampler_wrapper(method, incoming_sample_readers))

    # wrap the sample producer
    for f in component_class.handler_roles['sample_producer']:
        method = getattr(obj, f.__name__) # f would be just the method before the sampler wrap
        setattr(obj, f.__name__, sample_producer_wrapper(method, outgoing_sample_writers))


    ## all the triggers
    # the rpc (set up after the wrapping)
    for name, chan in incoming_rpcs.items():
        chan.start_message_handling_thread(getattr(obj, name))

    # loops 
    for f in component_class.handler_roles['loop']:
        t = create_thread(loop_func, func=getattr(obj, f.__name__), ideal_interval=loop_intervals[f.__name__])
        t.start()

class ComponentStarter:
    def __init__(
        self, 
        component_class:Type[ComponentInterface], 
        manager: SyncManager, 
        loop_intervals: Dict[str, float]={}, 
        instantiator=None, 
        init_kwargs={}, 
        sample_setup_kwargs = {}
        ):
        """

        """


        incoming_rpcs: Dict[str, CallChannel] = {}


        # create a message channel for each method with @rpc, identified by function name
        for f in component_class.handler_roles['rpc']:
            
            chan = CallChannel(f.__qualname__, manager, getattr(f, "args_reader"), getattr(f, "args_writer")) 
            incoming_rpcs[f.__name__] = chan

            
        #component_class.samplers
        if len(component_class.handler_roles['sample_producer']) > 1:
            raise NotImplementedError

        if len(component_class.handler_roles['sample_producer']): 
            sample_producer = component_class.handler_roles['sample_producer'][0]

            assert hasattr(sample_producer, "setup_func")
            setup_func = getattr(sample_producer, "setup_func")
            outgoing_sampler_readers, outgoing_sample_writers = setup_func(**sample_setup_kwargs)

        else: 
            outgoing_sampler_readers, outgoing_sample_writers = [], []



        # states 
        self.__component_class = component_class
        self.__instantiator = instantiator if instantiator else component_class
        self.__init_kwargs = init_kwargs
        self.__loop_intervals = loop_intervals

        self.incoming_rpcs = incoming_rpcs
        self.outgoing_sample_readers = outgoing_sampler_readers



        self.__outgoing_sample_writers = outgoing_sample_writers
        self.__outgoing_rpcs: Dict[str, CallChannel] = {}
        self.__incoming_sample_readers: List[SampleReader] =[]


        self.process_starter: Optional[ProcessStarter] = None


    @property
    def outgoing_samples(self):
        # alias
        return self.outgoing_sample_readers

    def register_outgoing_rpc(self, outgoing_rpcs: Dict[str, CallChannel]):
        self.__outgoing_rpcs = outgoing_rpcs


    def register_incoming_samples(self, incoming_samples: List[SampleReader]):
        self.__incoming_sample_readers = incoming_samples 

    def get_outgoing_sample_writers(self):
        return self.__outgoing_sample_writers


    def start(self, **kwargs):
        assert not self.process_starter, "already started"
        self.process_starter = ProcessStarter(self.__start)
        self.process_starter.start(**kwargs)

    def __start(self):

        process = create_process(
            target, 
            component_class=self.__component_class,
            instantiater=self.__instantiator,
            incoming_sample_readers= self.__incoming_sample_readers, 
            incoming_rpcs= self.incoming_rpcs, 
            outgoing_sample_writers= self.__outgoing_sample_writers, 
            outgoing_rpcs= self.__outgoing_rpcs, 
            loop_intervals = self.__loop_intervals, 
            init_kwargs= self.__init_kwargs
            )
        process.start()
        return process

    


class ProcessStarter:
    """
    should have named it process manager...
    """
    def __init__(self, starter: Callable[P, Process], /, *args: P.args, **kwargs: P.kwargs):
        self.__start = starter 
        self.args = args
        self.kwargs= kwargs
        self.process: Optional[Process] = None
        self.started = False
        self.last_attempt = False
        self.killed = False
        

    def start(self, retry: int=5, check_interval:float=2):

        assert not (self.started or self.killed)
        self.started = True

        self.process = self.__start(*self.args, **self.kwargs)

        # the retry thread also joins the process to make sure it doesn't end when the main process has nothing to run
        self.retry_thread = create_thread(self.__retry_n_times, retry, check_interval)
        self.retry_thread.start()

    def is_alive(self):
        """
        
        """
        return (not self.last_attempt) or self.is_process_alive()

    def is_process_alive(self):
        try: 
            return self.process.is_alive() # type: ignore
        except:
            # 1. process is closed
            # 2. process is None            
            return False

    @deprecated('old alias')
    def kill(self):
        """
        old alias
        """
        self.termainate()

    def termainate(self):
        
        # TODO: change is to self.terminated 
        self.killed = True
        if self.process: 
            try: 
                self.process.terminate()
                self.process.join()
                self.process.close()
            except:
                pass

    def __retry_n_times(self, n_times:int, check_interval:float):
        """
        TODO: 
        this may have unintended consequences
        the failed RPC call will never return or be fulfilled if it is not put back in queue 
        """
        nth_time = 0
        while (nth_time < n_times):
            while self.is_process_alive():
                time.sleep(check_interval)
                
            if self.process:
                self.process.close()

            if not self.killed: 
                self.process = self.__start(*self.args, **self.kwargs)

            nth_time += 1

        # no more retry
        self.last_attempt = True

        # to make sure the process doesn't just get finished when the main process has nothing to run
        if self.process:
            self.process.join()
            

    


from queue import Queue
# concurrent.futures.ThreadPoolExecutor shutdown after idling for awhile?
# https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
#       "it is recommended that ThreadPoolExecutor not be used for long-running tasks."
class ThreadHandler:
    def __init__(self):
        self.run = True
        self.queue: Queue[Tuple[Callable, Tuple, Dict]] = Queue()
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def call(self, _func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs):
        self.queue.put((_func, args, kwargs))

    def loop(self):
        while self.run:
            func, args, kwargs = self.queue.get()
            func(*args, **kwargs)
            
    def kill(self):
        self.run = False

