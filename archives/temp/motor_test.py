import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)

# Set the pin as output
GPIO.setup(12, GPIO.OUT)

# Create a PWM instance
pwm = GPIO.PWM(12, 1000) # 100 Hz frequency
pwm.start(50) # 50% duty cycle



try:
    while 1:
        cycle = int(input("duty cycle: \n"))
        pwm.ChangeDutyCycle(cycle)
        time.sleep(0.001)
        
except KeyboardInterrupt:
    # Clean up
    pwm.stop()
    GPIO.cleanup()

