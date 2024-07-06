from typing import List, Dict, Literal, Union

def speed_proportion_control(proportion, speed) -> (float, float, List):
    """
    balance*speed = left speed
    (1-balance)*speed = right speed
    """
    warnings = []
    if not (proportion >=0) and (proportion <= 1):
        warnings.append({'proportion': proportion})
        proportion = min(proportion, 1)
        proportion = max(proportion, 0)

    left = 2*speed*proportion
    right = 2*speed - left
    return left, right, warnings


def radius_steer_control(
    radius, 
    speed, 
    direction: Literal['right', 'left'], 
    wheel_distance,
    max_speed=100, 
    ) -> (float, float, List): 
    """
    one wheel is slower than the other at the same PWM. This can't be used

    
    radius defines the curve radius

    radius & wheel_distance have the same unit
    radius should be <= wheel_distance/2


    
    speed has an arbitary unit. 
    returning the speed for left and right wheels with the same unit

    """

    def calculate(radius, speed, direction, wheel_distance):
    
        # assume turning left 
        right_radius =  wheel_distance/2 + radius
        left_radius = right_radius - wheel_distance
        
        angular_speed = speed / radius
        right_speed = angular_speed * right_radius
        left_speed = angular_speed * left_radius
        
        if direction == 'right':
            left_speed, right_speed = right_speed, left_speed
        
        return left_speed, right_speed


    warnings = []

    if radius < wheel_distance/2:
        warnings.append({'radius': radius})
        radius = wheel_distance/2
    
    left_speed, right_speed = calculate(radius, speed, direction, wheel_distance)

    if left_speed > max_speed:
        warnings.append({'left_speed': left_speed})
        left_speed = max_speed
    
    if right_speed > max_speed:
        warnings.append({'right_speed': right_speed})
        right_speed = max_speed

    return left_speed, right_speed, warnings
        
        
    
