import RPi.GPIO as GPIO
class TwoWheels:
    def __init__(self, pwm_left: GPIO.PWM, pwm_right: GPIO.PWM):
        pwm_left.start(0)
        pwm_right.start(0)

        self.pwm_left = pwm_left
        self.pwm_right = pwm_right

    def control(self, left, right):
        self.pwm_left.ChangeDutyCycle(left)
        self.pwm_right.ChangeDutyCycle(right)

