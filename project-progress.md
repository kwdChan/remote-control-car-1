# Project progress
## Intro
- A car with two active wheels in the front and one passive wheel in the back
- Change to ratio of the motor power of the two wheels for steering
- It has a IMU to tell the car orientation and the angular velocity
- Goals:
  1. Make the car self-driving with the camera input
  2. To drive the car with the pitch of my sound (Competition Entry)

## Project Started: Late Jan 2024 
- The motors, wheels, the black base plate and the screws came together in a kit I brought online
- I got the other components separately. They are just *placed* on the car.
- The pitch control was not the goal at this point.
  - I just wanted to drive the car to train my neural network.
  - So you can see a infra-red receiver on the Pi
<img src="https://github.com/kwdChan/remote-control-car-1/assets/64915487/07172e8a-2179-4202-90a3-c640d354abc2" width="500" />

## 1. Feb  
- Infra-red control didn't work well. Decided to use other medium for the communication.
- Pitch detection was implemented. It kinda worked.
- I decided to make the car to be controlled by only the sound "Weeeee" but not any other pitches
  - The *plan* was to use a neural network to detect the word "we" to switch on the pitch control mode
 
https://github.com/kwdChan/remote-control-car-1/assets/64915487/0b0aa39c-59d6-4893-a01e-fe37b6e12d99


## 2. Mar 
- I knew about the makerspace around this time and got inducted.
- Printed the case for the Pi and the powerbank
- Laser cutted the camera holder (half succeeded)
<img src="https://github.com/kwdChan/remote-control-car-1/assets/64915487/49c26e7f-ce42-4057-b291-b6704ca331c2" width="300" />
<img src="https://github.com/kwdChan/remote-control-car-1/assets/64915487/eddaedbd-3d29-4831-842d-dbe7f6d9170d" width="300" />

## 3. Mar - Apr 
- I implemented bluetooth car control (for the self-driving goal) but I couldn't car the car to go straight  
- So I got the IMU and trying to make it work. Now the car can be controlled by specifying the angular velocity (degrees per second)
- But see one of my failed attempt:

https://github.com/kwdChan/remote-control-car-1/assets/64915487/feb3af07-c570-49aa-8b61-ae2d93b7aea3

## 4. May - Jun
- Made a few attempts for the self-driving but none of them worked at all
- Printed a better holder for the camera
- Attempted to attach a display to the car but I broke it (by forcing it to my 3D printed case)
- Did a lot of restructuring of the code to accelerate the development
<img src="https://github.com/kwdChan/remote-control-car-1/assets/64915487/00f6f5c8-7741-4374-bb8d-b31f5010892e" width="300" />

## Now 
- The deadline is approaching but I haven't done much for the pitch control
- I reimplemented the pitch control to my new car control code yesterday
- The pitch detection works well **as long as the motor doesn't move**.
- I new problem I didn't have back then. 
![Pasted image 20240706212048](https://github.com/kwdChan/remote-control-car-1/assets/64915487/640cfb1b-afb0-4572-9044-aeea9de51eaa)


