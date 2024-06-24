from components import ComponentInterface, component, loop, samples_producer, sampler, rpc, ComponentStarter, declare_method_handler
from multiprocessing import Manager

from components import numpy_sample_setup
import numpy as np 


@component
class MyTestComponent2(ComponentInterface):
    def __init__(self):
        self.idx = -1

    @loop
    @samples_producer(typecodes=['d', 'd'], default_values=[0, np.zeros((4,4))])
    @sampler
    def step(self, idx_other, arr_other):
        self.idx += 1
        print(idx_other)
        return self.idx, np.random.random((4,4)) 
        
    @rpc()
    def com2_rpc(self, x:int, y:str) -> int:

        print(x, y)

        return x+int(y)


def testing_function2():
    m = Manager()

    sample_reader, sample_writer = numpy_sample_setup(['d', 'd'], [0, np.zeros((4,4))])
    
    starter1 = ComponentStarter(
        MyTestComponent2, 
        manager=m, 
        init_kwargs={}, 
        loop_intervals=dict(step=1), 
        instantiator=None, 
        sample_setup_kwargs=dict(default_values=[0, np.zeros((4,4))])
    )

    starter1.register_incoming_samples(sample_reader)
    starter1.register_outgoing_rpc({})

    chan = starter1.incoming_rpcs['com2_rpc']
    
    chan = declare_method_handler(chan, MyTestComponent.com1_rpc)

    # chan.call(1, y='2')
    # chan.call(1, 2)
    # chan.call(c = 1, x=1, y='2')

    starter1.start()

    return starter1, sample_writer


@component
class MyTestComponent(ComponentInterface):
    def __init__(self):
        self.idx = -1

    @loop
    @samples_producer(typecodes=['d', 'd'])#default_values=[0, np.zeros((4,4))]
    @sampler
    def step(self, idx_other, arr_other):
        self.idx += 1
        print(idx_other)
        return self.idx, np.random.random((4,4)) 
        

    @rpc()
    def com1_rpc(self, x:int, y:str) -> int:

        print(x, y)

        return x+int(y)


def testing_function():
    m = Manager()

    sample_reader, sample_writer = numpy_sample_setup(['d', 'd'], [0, np.zeros((4,4))])
    
    starter1 = ComponentStarter(
        MyTestComponent, 
        manager=m, 
        init_kwargs={}, 
        loop_intervals=dict(step=1), 
        instantiator=None, 
        sample_setup_kwargs=dict(default_values=[0, np.zeros((4,4))])
    )

    starter1.register_incoming_samples(sample_reader)
    starter1.register_outgoing_rpc({})

    chan = starter1.incoming_rpcs['com1_rpc']
    
    chan = declare_method_handler(chan, MyTestComponent.com1_rpc)

    # chan.call(1, y='2')
    # chan.call(1, 2)
    # chan.call(c = 1, x=1, y='2')

    starter1.start()

    return starter1, sample_writer
