#from gpiozero import LED
import RPi.GPIO as GPIO
import time
from datetime import datetime, timedelta

# Use the GPIO numbers, not the pin numbers
GPIO.setmode(GPIO.BOARD) 

TRIG_PIN = 11
ECHO_PIN = 13

GPIO.setup(ECHO_PIN, GPIO.IN,  pull_up_down=GPIO.PUD_UP)
GPIO.setup(TRIG_PIN, GPIO.OUT)


# Setup the pin as an output
while True:
    GPIO.output(TRIG_PIN, GPIO.LOW)
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(20e-6)
    GPIO.output(TRIG_PIN, GPIO.LOW)

    while not GPIO.input(ECHO_PIN):
        pass

    start_time = time.monotonic()

    while GPIO.input(ECHO_PIN):
        pass

    time_diff = (time.monotonic() - start_time)
    print(time_diff*340/2)
    time.sleep(.1)
