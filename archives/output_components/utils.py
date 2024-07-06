import RPi.GPIO as GPIO
import warnings

GPIO.setmode(GPIO.BOARD)
warnings.warn("Using GPIO.setmode(GPIO.BOARD). Do not run BCM: This module assumes the GPIO.BOARD mode is used.")

def setup_pwm(pin, freq=300) -> GPIO.PWM:

    GPIO.setup(pin, GPIO.OUT)
    return GPIO.PWM(pin, freq) 