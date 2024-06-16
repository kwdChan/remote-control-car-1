from abc import abstractmethod
from re import L
import time
from typing import Callable, Tuple, TypeVar, get_origin, get_args, Union, Any, Optional, Dict, List
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

class MessageChannel:
    def __init__(self, name, manager: SyncManager):

        self.name = name
        self.manager = manager

        # states 
        self.msg_pending: ListProxy[Tuple[int, Any]] = manager.list()
        self.res_pending: DictProxy[int, Any] = manager.dict()

        
        self.response_locks: DictProxy[int, Any] = manager.dict()

        self.msg_id = -1
        self.msg_semaphore = Semaphore(0)


    def send_message(self, msg: Any, expect_response=False) -> Tuple[int, Optional[ThreahingLock]]:
        self.msg_id += 1 

        self.msg_pending.append((self.msg_id, msg))
        
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
        return msg_id, msg

    def reply(self, msg_id, res):
        l = self.response_locks.get(msg_id)
        if l is not None:
            self.res_pending[msg_id] = res
            l.release()

        else:
            pass
            #raise ValueError(f"a response is not expected for this msg_id: {msg_id}")

    def await_response(self, msg_id):
        self.response_locks[msg_id].acquire()
        return self.res_pending.pop(msg_id)

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

class EventBroadcaster:
    def __init__(self, name,  manager: SyncManager):
        self.name = name
        self.subscriptions: Dict[str, MessageChannel] = {}
        self.manager = manager

    def publish(self, msg):
        for name, sub in self.subscriptions.items():
            sub.send_message(msg, expect_response=False)

    def subscrible(self, receiver_name):
        new_sub = MessageChannel(receiver_name, manager=self.manager)
        self.subscriptions[receiver_name] = new_sub
        return new_sub


def loop(func, ideal_interval):
    while True:
        st = time.monotonic()
        func()
        et = time.monotonic()
        
        itv_to_sleep = ideal_interval-(et-st)
        if itv_to_sleep <= 0:
            warnings.warn("the real interval is >= ideal_interval")
        time.sleep(max(0, itv_to_sleep))

def component(events_to_produce: List[str]):
    """
    events_to_produce: 
        the name of the event should match the parameter of the init 
        the object will be a event broadcaster object 

    """
    def dec(cls):    
        assert not hasattr(cls, "rpc_list"), "reserved attribute name"
        assert not hasattr(cls, "event_handlers"), "reserved attribute name"
        assert not hasattr(cls, "events_to_produce"), "reserved attribute name"

        cls.rpc_list = []
        cls.event_handlers = {}
        cls.events_to_produce = events_to_produce

        for k, v in cls.__dict__.items():
            if not hasattr(v, "handler_type"): continue

            if v.handler_type == "event_handler":
                assert hasattr(v, "event_name")
                cls.event_handlers[v.event_name] = v
            
            elif v.handler_type == 'rpc':
                cls.rpc_list.append(v)

            else:
                raise NotImplementedError

        return cls
    return dec

def event_handler(event_name: str):
    """
    the MessageChannel is instantiated by the event broadcaster from other components 
    """

    def dec(func: Callable): 
        
        setattr(func, "handler_type", "event_handler")
        setattr(func, "event_name", event_name)

        return func

    return dec 

def rpc(func: Callable):
    """
    the MessageChannel is instantiated by this component and passed on to the other components
    """
    #TODO: type check the call to make sure it only takes one message 
    setattr(func, "handler_type", "rpc")

    return func 

    
def target(
    component_class, 
    instantiater: Callable, 
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

    func = get_handler_from_step(obj.step, incoming_value_samplers, outgoing_value_assigners)

    loop(func, **loop_kwargs)

    

def create_component_starter(component_class, manager, loop, init_kwargs, loop_kwargs, instantiater=None):

    assert hasattr(component_class, "rpc_list")
    assert hasattr(component_class, "event_handlers")
    assert hasattr(component_class, "events_to_produce")
    
    incoming_channels = {}
    incoming_rpc = {}

    for f in component_class.rpc_list:
        chan = MessageChannel(f.__qualname__, manager)
        incoming_channels[f.__name__] = chan
        incoming_rpc[f.__name__] = chan


    outgoing_event_broadcasters = {e: EventBroadcaster(e, manager) for e in component_class.events_to_produce}

    outgoing_value_samplers, outgoing_value_assigners = component_class.shared_samples(manager)


    # incoming_channels(incoming_rpc only), outgoing_value_assigners, outgoing_event_broadcasters, 
    # incoming_value_samplers, outgoing_rpc , incoming_events 

    def starter(incoming_value_samplers, outgoing_rpc, incoming_events: Dict[str, EventBroadcaster]):

        for event_name, f in component_class.event_handlers.items():
            incoming_channels[f.__name__] = incoming_events[event_name].subscrible(f.__qualname__)

        process = Process(
            target=target, 
            kwargs=dict(
                component_class = component_class, 
                instantiater = instantiater if instantiater else component_class, 
                incoming_value_samplers=incoming_value_samplers, 
                incoming_channels=incoming_channels, 
                outgoing_value_assigners=outgoing_value_assigners, 
                outgoing_rpcs=outgoing_rpc, 
                outgoing_event_broadcasters=outgoing_event_broadcasters, 
                loop=loop, 
                loop_kwargs=loop_kwargs, 
                init_kwargs=init_kwargs,
                )
            )
        process.start()
        return process

    return starter, outgoing_value_samplers, incoming_rpc, outgoing_event_broadcasters




# def register_event_handlings_v2(channels: List[MessageChannel], handlers:List[Callable]):
#     threads = []
#     for c, h in zip(channels, handlers):
#         threads.append(c.start_message_handling_thread(h, False))

#     return threads  


# def register_event_handlings(channels: List[MessageChannel], handlers:List[Callable]):
#     threads = []
#     for c, h in zip(channels, handlers):
#         t = threading.Thread(target=c.message_handling_loop, kwargs=dict(handler=h))
#         threads.append(t)

#     for t in threads: 
#         t.start()
#     return threads 

    






# # make this into an interface 
# class Component: 

#     logger: Logger
#     shared_values: List[BaseProxy]
    

#     @abstractmethod
#     def step(self) -> Tuple:
#         # have to return tuple because the main will iterate through the output variables 
#         return ()




#     @classmethod
#     def __not_used(cls, manager: BaseManager) -> List[Optional[BaseProxy]]:
#         """
#         override this method to set the ctypes and initial values for the shared values 
#         use the type hint to infer by default 
#         """ 
#         return [None]
#         raise NotImplementedError("This method needs to be overriden")


#         step_return_types = inspect.signature(cls.step).return_annotation
        
#         if get_origin(step_return_types) is tuple:
#             # e.g.: func() -> Tuple[str, int]
#             output_types = get_args(step_return_types)
#         else: 
#             # e.g. func() -> str 
#             output_types = (step_return_types, )

#         shared_outputs = []
#         for o in output_types:
#             if not o in cls.INFERRED_TYPES:
#                 raise ValueError(f"The provided type:{o} is cannot be inferred. Override this method.")
#             shared_outputs.append(Value(*cls.INFERRED_TYPES[o]))

#         return shared_outputs


