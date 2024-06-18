from abc import abstractmethod
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



class MessageChannel:
    
    @staticmethod
    def default_msg_reader(msg):
        return msg

    @staticmethod
    def default_msg_writer(store):
        return store

    def __init__(
        self, 
        name, 
        manager: SyncManager, 
        msg_reader: Optional[Callable[[Any], Any]]=None, 
        msg_writer: Optional[Callable[[Any], Any]]=None, 
    ):

        self.name = name
        self.manager = manager

        self.msg_reader = msg_reader if msg_reader else MessageChannel.default_msg_reader 
        self.msg_writer = msg_writer if msg_writer else MessageChannel.default_msg_writer

        # states 
        self.msg_pending: ListProxy[Tuple[int, Any]] = manager.list()
        self.res_pending: DictProxy[int, Any] = manager.dict()

        
        self.response_locks: DictProxy[int, Any] = manager.dict()

        self.msg_id = -1
        self.msg_semaphore = Semaphore(0)


    def send_message(self, msg: Any, expect_response=False) -> Tuple[int, Optional[ThreahingLock]]:
        self.msg_id += 1 

        self.msg_pending.append((self.msg_id, self.msg_writer(msg)))
        
        lock = None
        if expect_response: 
            lock = self.manager.Lock()
            lock.acquire()
            self.response_locks[self.msg_id] = lock

        # CANNOT RELEASE THE LOCK BEFORE THE self.response_locks[self.msg_id] IS SET
        self.msg_semaphore.release()
        return self.msg_id, lock
        
    def await_message(self):
        self.msg_semaphore.acquire()
        msg_id, msg = self.msg_pending.pop(0)
        return msg_id, self.msg_reader(msg)

    def reply(self, msg_id, res):
        l = self.response_locks.get(msg_id)
        if l is not None:
            self.res_pending[msg_id] = self.msg_writer(res)
            l.release()

        else:
            pass
            #raise ValueError(f"a response is not expected for this msg_id: {msg_id}")

    def await_response(self, msg_id):
        self.response_locks[msg_id].acquire()
        return self.msg_reader(self.res_pending.pop(msg_id))

    def await_and_handle_message(self, handler: Callable[[Any], Any]):
        msg_id, msg = self.await_message()
        res = handler(msg)
        self.reply(msg_id, res)
        return res

    def message_handling_loop(self, handler: Callable[[Any], Any]):
        while True:
            self.await_and_handle_message(handler)

    def start_message_handling_thread(self, handler: Callable[[Any], Any]):
        # TODO: saving t to self.t may make the object unpickleable?
        # TODO: have a pool of threads to consume the messages as fast as possible 
        t = threading.Thread(target=self.message_handling_loop, kwargs=dict(handler=handler))
        t.start()
        return t
        
    def send_and_await_response(self, msg):
        msg_id, lock = self.send_message(msg, expect_response=True)
        return self.await_response(msg_id)


class EventBroadcaster:
    def __init__(self, name,  manager: SyncManager, msg_reader=None, msg_writer=None):
        self.name = name
        self.subscriptions: Dict[str, MessageChannel] = {}
        self.manager = manager
        self.msg_reader = msg_reader
        self.msg_writer = msg_writer

        self.locked = False


    def publish(self, msg):
        for name, sub in self.subscriptions.items():
            sub.send_message(msg, expect_response=False)

    def subscrible(self, receiver_name):
        assert not self.locked, "locked. the process might have started already"

        new_sub = MessageChannel(
            receiver_name, 
            manager=self.manager, 
            msg_reader = self.msg_reader, 
            msg_writer = self.msg_writer, 
            )
        self.subscriptions[receiver_name] = new_sub
        return new_sub

    def lock(self):

        self.locked = True

def get_handler_from_step(step: Callable, input_readers, output_assigners):
    """
    the parameters are sampled and the step is called
    the returned values of the step is used to set the values to be sampled by other processes 
    """

    def handler(msg=None): 
        outputs = step(*[f() for f in input_readers])
        if outputs is None: 
            outputs = ()
        for idx, o in enumerate(outputs): 
            output_assigners[idx](o)
    return handler 


def loop(func, ideal_interval):
    while True:
        st = time.monotonic()
        func()
        et = time.monotonic()
        
        itv_to_sleep = ideal_interval-(et-st)
        if itv_to_sleep <= 0:
            warnings.warn("the real interval is >= ideal_interval")
        time.sleep(max(0, itv_to_sleep))


component_decorator_param_type = Dict[str, Optional[Tuple[Callable[[Any], Any], Callable[[Any], Any]]]]

class ComponentInterface:
    rpc_list = []
    event_handlers = {}
    samplers = []
    sample_producers = []
    events_to_produce: component_decorator_param_type = {}

ComponentSubtype = TypeVar('ComponentSubtype', bound=ComponentInterface)

def component(events_to_produce: component_decorator_param_type):
    
    """
    TODO: use roles to unify all the handler types 
    events_to_produce: 
        the name of the event should match the parameter of the init 
        the object will be a event broadcaster object 

    """
    
    def dec(cls: Type[ComponentSubtype]):    
        # assert not hasattr(cls, "rpc_list"), "reserved attribute name"
        # assert not hasattr(cls, "event_handlers"), "reserved attribute name"
        # assert not hasattr(cls, "events_to_produce"), "reserved attribute name"
        # assert not hasattr(cls, "samplers"), "reserved attribute name"
        # assert not hasattr(cls, "sample_producers"), "reserved attribute name"

        cls.rpc_list = []
        cls.event_handlers = {}
        
        cls.samplers = []
        cls.sample_producers = []
        
        cls.events_to_produce = events_to_produce

        for k, v in cls.__dict__.items():
            if not hasattr(v, "handler_type"): continue

            if v.handler_type == "event_handler":
                assert hasattr(v, "event_name")
                cls.event_handlers[v.event_name] = v
            
            elif v.handler_type == 'rpc':
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


def event_handler(event_name: str):
    """
    the MessageChannel is instantiated by the event broadcaster from other components 
    """

    def dec(func: Callable[[Any, Any], Any]): 
        
        setattr(func, "handler_type", "event_handler")
        setattr(func, "event_name", event_name)

        return func

    return dec 

def rpc(
    msg_reader: Optional[Callable[[Any], Any]] = None, 
    msg_writer: Optional[Callable[[Any], Any]] = None
) -> Callable[[Callable], Callable]:
    
    def dec(func: Callable[[Any, Any], Any]):
        """
        the MessageChannel is instantiated by this component and passed on to the other components
        """
        setattr(func, "handler_type", "rpc")
        setattr(func, "msg_reader", msg_reader)
        setattr(func, "msg_writer", msg_writer)

        return func 

    return dec



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



SampleReader = Callable[[], Any]
SampleWriter = Callable[[Any], None]

SampleSetupFunction = Callable[[List[Any], List[Any]], Tuple[List[SampleReader], List[SampleWriter]]]

def default_sample_setup(typecodes, default_values): 
    readers, writers = [], []
    for tc, dv in zip(typecodes, default_values):
        r, w = shared_value_v2(tc, dv)
        readers.append(r)
        writers.append(w)

    return readers, writers


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


def samples_producer(typecodes:List[Any], default_values:List[Any], setup_function:SampleSetupFunction=numpy_sample_setup):

    def setup(): 
        return setup_function(typecodes, default_values)
        
    def dec(func: Callable): 

        setattr(func, "handler_type", "sample-related")
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
    setattr(func, "handler_type", "sample-related")

    if not hasattr(func, "roles"):
        setattr(func, "roles", [])
    getattr(func, "roles").append("sampler")
    
    return func

def target(
    component_class: Type[ComponentInterface], 
    instantiater: Callable[..., ComponentInterface], 
    incoming_value_samplers: List[Callable[[], Any]], 
    incoming_channels:Dict[str, MessageChannel], 
    outgoing_value_assigners: List[Callable[[Any], None]],
    outgoing_rpcs:Dict[str, MessageChannel], 
    outgoing_event_broadcasters:Dict[str, EventBroadcaster], 
    loop: Callable[[Callable], None], 
    loop_kwargs, 
    init_kwargs, 
    ):

    obj = instantiater(**init_kwargs, **outgoing_rpcs, **outgoing_event_broadcasters)

    for name, chan in incoming_channels.items():
        chan.start_message_handling_thread(getattr(obj, name))


    sampler_names = [s.__name__ for s in component_class.samplers]
    sample_producer_names = [s.__name__ for s in component_class.sample_producers]

    if len(sample_producer_names) > 1:
        raise NotImplementedError("having multiple sample produced is not implemented")
    
    if len(sampler_names) > 1:
        raise NotImplementedError("having multiple sampler is not implemented")
    

    sampler_handlers = []
    for name in sampler_names:
        sampler_handlers.append(
            get_handler_from_step(
                getattr(obj, name), 
                incoming_value_samplers, 
                outgoing_value_assigners, 
                )

        )
    
    loop(sampler_handlers[0], **loop_kwargs)




def create_component_starter(
    component_class:Type[ComponentInterface], 
    manager: SyncManager, 
    loop, init_kwargs, loop_kwargs, instantiater=None):

    assert hasattr(component_class, "rpc_list")
    assert hasattr(component_class, "event_handlers")
    assert hasattr(component_class, "events_to_produce")
    
    incoming_channels = {}
    incoming_rpc = {}

    for f in component_class.rpc_list:
        
        chan = MessageChannel(f.__qualname__, manager, f.msg_reader, f.msg_writer) 
        incoming_channels[f.__name__] = chan
        incoming_rpc[f.__name__] = chan

    
    events_to_produce = cast(component_decorator_param_type, component_class.events_to_produce)

    outgoing_event_broadcasters = {}
    for event_name, reader_writer in events_to_produce.items():
        reader_writer = () if reader_writer is None else reader_writer
        outgoing_event_broadcasters[event_name] = EventBroadcaster(event_name, manager, *reader_writer)

    
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
        def __init__(self, outgoing_value_samplers, incoming_rpc, outgoing_event_broadcaster) :
            self.process: Process


            self.incoming_rpc: Dict[str, MessageChannel] = incoming_rpc
            self.outgoing_samples:List[SampleReader] = outgoing_value_samplers
            self.outgoing_events: Dict[str, EventBroadcaster] = outgoing_event_broadcaster


            self.outgoing_rpc: Dict[str, MessageChannel] = {}
            self.incoming_samples:List[SampleReader] = []
            self.incoming_events: Dict[str, EventBroadcaster] = {}

        def start(self):

            for k, v in self.outgoing_events.items():
                v.lock()


            self.process = Process(
            target=target, 
            kwargs=dict(
                component_class = component_class, 
                instantiater = instantiater if instantiater else component_class, 
                incoming_value_samplers=self.incoming_samples, 

                incoming_channels=incoming_channels, 
                outgoing_value_assigners=outgoing_value_assigners, 
                outgoing_rpcs=self.outgoing_rpc, 

                outgoing_event_broadcasters=outgoing_event_broadcasters, 
                loop=loop, 
                loop_kwargs=loop_kwargs, 
                init_kwargs=init_kwargs,
                )
            )
            self.process.start()


        def register_incoming_events(self, incoming_events: Dict[str, EventBroadcaster]):
            # TODO: this is very cursed 
            for event_name, f in component_class.event_handlers.items():
                incoming_channels[f.__name__] = incoming_events[event_name].subscrible(f.__qualname__)

        def register_outgoing_rpc(self, outgoing_rpc):
            self.outgoing_rpc = outgoing_rpc


        def register_incoming_samples(self, incoming_samples: List[SampleReader]):
            self.incoming_samples = incoming_samples 

            
    
    starter = ComponentStarter(outgoing_value_samplers, incoming_rpc, outgoing_event_broadcasters)


    return starter



@component({"event_from_com1": None})
class MyTestComponent(ComponentInterface):
    def __init__(self, event_from_com1: EventBroadcaster):
        self.event_from_com1 = event_from_com1
        self.idx = -1

    @samples_producer(['d', 'd'], [0, np.zeros((4,4))])
    @sampler
    def step(self, idx_other, arr_other):
        self.idx += 1

        self.event_from_com1.publish({"step": self.idx})
        
        
        return self.idx, np.random.random((4,4))
        

    @event_handler("increment")
    def recv(self, msg):
        print(f'increment: {msg}')
        self.idx += msg


    @rpc()
    def com1_rpc(self, msg):

        print(f'com1 received msg: {msg}', flush=True)

        return msg


def testing_function():
    m = Manager()

    sample_reader, sample_writer = numpy_sample_setup(['d', 'd'], [0, np.zeros((4,4))])
    

    event = EventBroadcaster('increment_from_outside', m)

    starter1 = create_component_starter(

        MyTestComponent, manager=m, loop=loop, init_kwargs={}, loop_kwargs={'ideal_interval': 1}, instantiater=None
    )

    starter1.register_incoming_samples(sample_reader)
    starter1.register_outgoing_rpc({})
    starter1.register_incoming_events({"increment":event})
    sub = starter1.outgoing_events['event_from_com1'].subscrible('from_outside')
    
    starter1.start()

    return starter1, sample_writer, event, sub


