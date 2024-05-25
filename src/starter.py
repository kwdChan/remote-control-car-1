from multiprocessing import Process
import subprocess, sys, time, datetime, atexit, signal

LOG_PATH = '/home/kawa/projects/car1/log/'
def start():
    return subprocess.Popen([sys.executable, '/home/kawa/projects/car1/src/main.py', LOG_PATH+str(datetime.datetime.now())])



PROCESS = start()

def on_exit(signum=None, frame=None):
    PROCESS.terminate()
    sys.exit(0)

atexit.register(PROCESS.terminate)
signal.signal(signal.SIGTERM, on_exit)


def main(): 
    global PROCESS
    while True: 
        if not PROCESS.poll(): 
            time.sleep(1)
        else:
            PROCESS = start()

main()
