import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BOARD)

PIN = 8

# Set the pin as output
GPIO.setup(PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

value = 0
GAMMA = 1/5000

try:
    while 1:
        value = GAMMA*GPIO.input(PIN) + (1-GAMMA)*value

        print(f"{value:.2f}", end='  \r')
        
        
except KeyboardInterrupt:
    # Clean up
    GPIO.cleanup()

