import subprocess, sys, time, datetime

LOG_PATH = './log/'
def main(): 
    def start():
        return subprocess.Popen([sys.executable, 'src/main.py', LOG_PATH+str(datetime.datetime.now())])
    
    p = start()
    while True: 
        if not p.poll(): 
            time.sleep(1)
        else:
            p = start()

main()