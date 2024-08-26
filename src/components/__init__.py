
try: 
    from components.controller.bluetooth_controller import *
    from components.controller.bluetooth_SPP_server import *
    from components.gyroscope.gyroscope import *
    from components.camera import *
    from components.logger import *
    from components.two_wheels import *
    from components.controller.image_ml import *


    from components.microphone.ahhhhh import *
    from components.microphone.microphone import *

except:
    print('some module failed')

from components.syncronisation import *
