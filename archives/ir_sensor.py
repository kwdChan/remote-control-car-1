from typing_extensions import deprecated
import RPi.GPIO as GPIO
import time
from typing import List, Dict, Literal, Union
import datetime
import numpy as np
from data_collection.data_collection import LoggerSet, Logger


from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
code2text = {
    16738455: '0',
    16750695: '100+',
    16756815: '200+',
    16724175: '1',
    16718055: '2',
    16743045: '3',
    16716015: '4',
    16726215: '5',
    16734885: '6',
    16769055: '-',
    16754775: '+',
    16720605: '|<<', 
    16712445: '>>|', 
    16761405: '>||',
    16748655: 'EQ'
}


@deprecated("Don't use IR remotes for car control")
class IRreciever:
    """
    Dev note:
        IR isn't good for remote car control
        If IR were to be used, pigpio should be used for a more precise timing
    """
    def __init__(self, pin:int, logger:Logger, n_sample=2000, interval=5e-6):
        
        GPIO.setmode(GPIO.BOARD) # type: ignore
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP) # type: ignore
        
        self.pin = pin 
        self.n_sample = n_sample
        self.interval = interval
        self.logger = logger

        # stateful
        self.last_code: int = 0

    def poll(self) -> int: 
        """
        blocking until a signal is recieved


        Dev note:
            Don't use the callback pattern.
            This forces the higher level components to also use callbacks. 
            And callback chaining is not desirable. 
        """
        
        sig = trigger_read(self.pin, self.n_sample, interval=self.interval)  
        code = to_number(sig2bits(sig))
        
        self.logger.log('sig', sig)
        self.logger.log('code', code)
        self.logger.log_time('time-sent')
        if code <= 1:
            return self.last_code
        else:
            self.last_code = code
            return code

    @staticmethod
    def main(pin:int, logger: Logger,  output_con: Connection):
        component = IRreciever(pin, logger)

        while True:
            logger.increment_idx()
            code = component.poll()
            output_con.send(code)

    @staticmethod  
    def start(pin:int, logger_set: LoggerSet, **kwargs):

        logger = logger_set.get_logger(**kwargs)
        receiver, sender = Pipe(False)
        process = Process(target=IRreciever.main, args=(pin, logger, sender))
        process.start()
        return process, receiver

#@deprecated("use pigpio")
def trigger_read(pin, n_sample, interval):
    """

    
    5e-6 & 1e-5 made no difference
    
    perhaps to sampling rate is limited

    the timing can sometimes be off too. 
    perhaps python is unfit for this function....
    """
    while GPIO.input(pin):
        pass

    samples = np.zeros(n_sample)
    
    for i in range(n_sample):
        samples[i] = GPIO.input(pin)
        time.sleep(interval)
    return samples

#@deprecated("use pigpio")
def sig2bits(sig:np.ndarray, zero_maxlen=15, one_maxlen=50, max_noise_length=1) -> List:
    """
    sometimes the signal be as short as 2 samples
    
    noise tolerant testing 
    result[::6] = 0
    result[3::6] = 1

    sig2bits(result[1:])    
    """
    search_for = 1

    def find_next(sig, value):
        """
        return:
            the remaining signal where the value is found and onwards
            the index 
            
        
        """
        idx = -1
        for idx, v in enumerate(sig):
            if v == value:
                return sig[idx:], idx
        # if not found
        return np.array([]), idx

    def find_next_noise_tolerant(sig, value, max_noise_length=max_noise_length, offset=0):
        """
        0000000010001000001111111
        ignore the 1 in the middle 

        000011000000100000000111111
        11000000100000000111111, idx=4, i=2   (6+0)
        100000000111111, idx=6, i=1   (6+1+6)
        111111, idx=8   (8+13)

        6 + 7 + 8 = 21
        """
        sig, idx = find_next(sig, value)
        if not len(sig):
            return sig, idx
        
        for i in range(max_noise_length):
            if not sig[i] == value:
                # 000011^0000111111
                return find_next_noise_tolerant(sig[i:], value, max_noise_length, offset=idx+i+offset)
        
        # the value length is long enough
        return sig, idx+offset
            
    bits = []
    if not (len(sig) and (sig[0]==0)):
        return bits

    # 0s -> 1s -> 0 burst -> signal

    # 0s -> 1s
    sig, idx = find_next_noise_tolerant(sig, 1)
    if not (len(sig)):
        return bits
    
    # 1s -> 0 burst
    sig, idx = find_next_noise_tolerant(sig, 0)

    if not (len(sig)):
        return bits

    bits = []
    while len(sig):
        
        # 0 burst -> 1 (signal)
        sig, length = find_next_noise_tolerant(sig, 1)
        if not (len(sig)):
            return bits

        
        # 1 (signal) -> 0 burst 
        sig, length = find_next_noise_tolerant(sig, 0)

        if length > one_maxlen: 
            return bits 

        if length > zero_maxlen:
            
            bits.append(1)
        else:
            bits.append(0)

    return bits 
    
#@deprecated("use pigpio")
def to_number(bits: List):
    value = 0
    for bit_pos, v in enumerate(bits[::-1]):
        value += v*(2**bit_pos)

    return value