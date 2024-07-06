import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)

# Set the pin as output
GPIO.setup(12, GPIO.OUT)

# Create a PWM instance
freq = 100
pwm = GPIO.PWM(12, freq) # 100 Hz frequency

dc = 15
pwm.start(dc) # 50% duty cycle

try:
    while 1:
        #pwm.ChangeFrequency(freq)

        dc = input('duty cycle percentage (0-100):')
        print(f"{float(dc)*freq/1000}ms")
        pwm.ChangeDutyCycle(float(dc))
        
        
except KeyboardInterrupt:
    # Clean up
    pwm.stop()
    GPIO.cleanup()

