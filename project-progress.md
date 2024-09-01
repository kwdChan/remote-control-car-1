# Project progress
## Intro
- A car with two active wheels in the front and one passive wheel in the back
- Change to ratio of the motor power of the two wheels for steering
- It has a IMU to tell the car orientation and the angular velocity
- Goals:
  1. Make the car self-driving with the camera input
  2. To drive the car with the pitch of my sound (Competition Entry)

## Late Jan 2024 - Project started 
- The motors, wheels, the black base plate and the screws came together in a kit I brought online
- I got the other components separately. They are just *placed* on the car.
- The pitch control was not the goal at this point.
  - I just wanted to drive the car to train my neural network.
  - So you can see a infra-red receiver on the Pi
<img src="https://github.com/kwdChan/remote-control-car-1/assets/64915487/07172e8a-2179-4202-90a3-c640d354abc2" width="500" />

## 1. Feb - Project goal decided
- Infra-red control didn't work well. Decided to use other medium for the communication.
- Pitch detection was implemented. It kinda worked.
- **I decided to make the car to be controlled by only the sound "Weeeee" but not any other pitches**
  - The plan was to use a neural network to detect the word "we" to switch on the pitch control mode
 
https://github.com/kwdChan/remote-control-car-1/assets/64915487/0b0aa39c-59d6-4893-a01e-fe37b6e12d99


## 2. Mar - 3D printing for the first time
- I knew about the makerspace around this time and got inducted.
- Printed the case for the Pi and the powerbank
- Laser cutted the camera holder (half succeeded)
<img src="https://github.com/kwdChan/remote-control-car-1/assets/64915487/49c26e7f-ce42-4057-b291-b6704ca331c2" width="300" />
<img src="https://github.com/kwdChan/remote-control-car-1/assets/64915487/eddaedbd-3d29-4831-842d-dbe7f6d9170d" width="300" />

## 3. Mar/Apr - The IMU Challenge
- I implemented bluetooth car control (for the self-driving goal) but I couldn't car the car to go straight  
- So I got the IMU and trying to make it work. Now the car can be controlled by specifying the angular velocity (degrees per second)
- But see one of my failed attempt:

https://github.com/kwdChan/remote-control-car-1/assets/64915487/feb3af07-c570-49aa-8b61-ae2d93b7aea3

## 4. May/Jun - Working on the self-driving goal
- Made a few attempts for the self-driving but none of them worked at all
- Printed a better holder for the camera
- Attempted to attach a display to the car but I broke it (by forcing it to my 3D printed case)
- Did a lot of restructuring of the code to accelerate the development
<img src="https://github.com/kwdChan/remote-control-car-1/assets/64915487/00f6f5c8-7741-4374-bb8d-b31f5010892e" width="300" />

eadline is approaching so I started working on the pitch control

## 5. Jul - We-recogition Model 
The deadline is approaching so I started working on the pitch control
The goal is to recognise the word "we" and then trigger the pitch detection

#### Method
- Common Voice Data (the audio recording of people speaking sentences + the sentences in text)
- Used an existing text-recording alignment model to find the timing of the word "we"
- Made the we-recordings into 0.4 second clips as the positive samples
- The negative samples are the randomly croped recording, also 0.4 second long
- A small convolution neural network was trained using pytorch, exported as ONNX

#### Result
##### The good part: 
- ~97% accuracy on the validation 
- The missed samples (false-negative) didn't sound like "we" to me too
- The false-positive samples sounded somewhat like "we" to me too
- The model worked on the my voice recorded from my mic
- The model runs fast enough to not cause a noticeable delay

##### The not-so-good part:  
- The model seemed to only detect "ee" or "e". It doesn't care about the "w"
- So "ki", "bee" "see", "se", "ke" all triggers the model
- The model get a bit excited by the car running noise
- ![Screenshot from 2024-09-01 10-52-28](https://github.com/user-attachments/assets/bf821497-4bb1-49d0-9093-b670591b4adf)

#### The next version
- I found an ambient noise dataset (ESC-50)
- The noise data will be mixed in to the recording

## 6. Aug - Pitch detection under noise 
- I reimplemented the pitch control to my new car control code in late Jun
- The pitch detection works well **as long as the motor doesn't move**.
- I new problem I didn't have back then. 
![Pasted image 20240706212048](https://github.com/kwdChan/remote-control-car-1/assets/64915487/640cfb1b-afb0-4572-9044-aeea9de51eaa)
- Collected the data (noise-only, noise+weeee, weeee-only) and started to find a way to detect the pitch under high noise 
- I could hear the weee myself under high noise so I was sure it's possible

### What I tried 
#### Add a few more microphones and do some magic on it
- Turns out Raspiberry Pi really doesn't have an ADC - the 3.5mm jack can't take microphone inputs
- So I only had to USB microphones very close to each other
- Magic 1 - beamforming: Spent some time reading about it and realised it's impossible for this separation 
- Magic 2 - ICA: Two mics only gives two sources. And the two microphones were too close to each other and I decided not to bother testing it out 
<img src="https://github.com/user-attachments/assets/42afeb41-c59f-4475-9515-78bc6b192a10" width="300" />

#### Use an AI to detect the pitch
- There's no eeeeeee data available so I had to synthesise my own 
- Spent me wayyyy too much time to reading about the synthesis (formant synthesis) with very limited success
- Also tried extending the "we" from the common voice data and it also didn't work well 
- I tried training a prove of concept model and it didn't work
- I decided this is out of my current capability 

#### Have the microphone remotely 
- Remote microphone doesn't seem to exist
- Use Pi Pico which does have an ADC so it was my backup option when I was trying the AI method 
- After some digging, ADC sampling at such high rate requires DMA which I have never done
- But I have no time for learning now (one week before deadline)

#### USB extension cable
Finally I bought an USB extension cable, which separated the mic from the car. 
The cable arrived this Monday and the pitch detection was good enough but it stills 

## This week 
- 3d printed a case 
<img src="https://github.com/user-attachments/assets/1ddf4e48-bfc8-4fdf-a71d-6967e3eaee33" width="300" />
<img src="https://github.com/user-attachments/assets/a6887057-16f0-465b-98fe-e08a843ad5f0" width="400" />





