#from gpiozero import LED
import RPi.GPIO as GPIO
import time
from datetime import datetime, timedelta
from uniplot import plot
from smbus2 import SMBus


# I2C channel 1 is connected to the GPIO pins
CHAN = 1
bus = SMBus(CHAN)

DEVICE_ADDR = 0x39


class REGISTER:
    # Bits: reserved, GEN, PIEN, AIEN, WEN, PEN, AEN, PON
    ENABLE = 0x80

    # ALS Gain Control<0:1>
    # Proximity Gain Control<3:2>
    CONTROL = 0x8F

    # Gesture Gain Control<6:5>
    # Gesture LED Drive Strength<4:3>
    # Gesture Wait Time<2:0>
    GCONFIG2 = 0xA3

    # GMODE: Gesture Mode<0>
    # GIEN: Gesture Interrupt Enable<1>
    # GFIFO_CLR: Gesture FIFO Clear<2> 
    GCONFIG4 = 0xAB

    # Gesture Pulse Count and Length Register (0xA6)
    # GPLEN 7:6
    # GPULSE 5:0
    GPULSE = 0xA6 

    PDATA = 0x9C # Proximity Data

    ATIME = 0x81  # ALS ADC intergation time 

    GFLVL = 0xAE # Gesture FIFO Level
    GFIFO_U = 0xFC # Gesture FIFO Data, UP, also that start of page read
    GFIFO_D = 0xFD # Gesture FIFO Data, DOWN
    GFIFO_L = 0xFE # Gesture FIFO Data, LEFT
    GFIFO_R = 0xFF # Gesture FIFO Data, RIGHT


    

# https://pypi.org/project/smbus2/

#read_byte_data
#read_i2c_block_data
#write_byte_data
#write_i2c_block_data



hist0 = [0 for _ in range(500)]
hist1 = [0 for _ in range(500)]
hist2 = [0 for _ in range(500)]
hist3 = [0 for _ in range(500)]

# Create a sawtooth wave 16 times
bus.write_byte_data(DEVICE_ADDR, REGISTER.ENABLE, 0b0100_0101)
bus.write_byte_data(DEVICE_ADDR, REGISTER.ATIME, 0xFF)
bus.write_byte_data(DEVICE_ADDR, REGISTER.CONTROL, 0x0F)
bus.write_byte_data(DEVICE_ADDR, REGISTER.GCONFIG4, 0x01)
bus.write_byte_data(DEVICE_ADDR, REGISTER.GCONFIG2, 0b01100000)
bus.write_byte_data(DEVICE_ADDR, REGISTER.GPULSE, 0x42)



def read_all():
    for _ in range(32):

        data = bus.read_i2c_block_data(DEVICE_ADDR, REGISTER.GFIFO_U, 4)


        hist0.append(data[0])
        hist1.append(data[1])

    plot([hist0[-100:], hist1[-100:], hist2[-100:], hist3[-100:]], y_min=0, y_max=255)
    

while 1:

    fifo_level = bus.read_byte_data(DEVICE_ADDR, REGISTER.GFLVL)
    if fifo_level == 32:
        read_all()
