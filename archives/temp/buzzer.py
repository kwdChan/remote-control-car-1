import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)

# Set the pin as output
GPIO.setup(12, GPIO.OUT)

# Create a PWM instance
pwm = GPIO.PWM(12, 1000) # 100 Hz frequency
pwm.start(5) # 50% duty cycle

freq = 1

try:
    while 1:
        pwm.ChangeFrequency(freq)
        freq+=1
        time.sleep(0.01)
except KeyboardInterrupt:
    # Clean up
    pwm.stop()
    GPIO.cleanup()

