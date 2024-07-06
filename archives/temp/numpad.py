#from gpiozero import LED
import RPi.GPIO as GPIO
import time

# Use the GPIO numbers, not the pin numbers
GPIO.setmode(GPIO.BOARD) 


SDA_PIN = 3
SCL_PIN = 5


GPIO.setup(SDA_PIN, GPIO.IN)
GPIO.setup(SCL_PIN, GPIO.OUT)

# Setup the pin as an output
try:
    while True:
        # Read the state of the pin
        GPIO.output(SCL_PIN, GPIO.HIGH)
        time.sleep(1.75/1000)
        for i in range(1, 5):
            GPIO.output(SCL_PIN, GPIO.LOW)
            GPIO.output(SCL_PIN, GPIO.HIGH)
            
            input_state = GPIO.input(SDA_PIN)

            # Print the state of the pin
            print(f"{i}:{input_state}  ", end='')
        print("", end='\n')


except KeyboardInterrupt:
    # Clean up GPIO on CTRL+C exit
    GPIO.cleanup()