from multiprocessing import Process
import subprocess, sys, time, datetime, atexit

LOG_PATH = '/home/kawa/projects/car1/log/'
def start():
    return subprocess.Popen([sys.executable, '/home/kawa/projects/car1/src/main.py', LOG_PATH+str(datetime.datetime.now())])



PROCESS = start()


@atexit.register
def terminate():
    PROCESS.terminate()

def main(): 
    global PROCESS
    while True: 
        if not PROCESS.poll(): 
            time.sleep(1)
        else:
            PROCESS = start()

main()
