import os
import threading
import multiprocessing as mp
import time
import queue


_QUEUE_COMMAND = queue.Queue()
_MAX_PROCESS = 1
_RLOCK = threading.RLock()
_STOP = False
_NB_EXECUTED = 0


def register_command(cmd):
    _QUEUE_COMMAND.put(cmd)

def get_max_processes():
    return _MAX_PROCESS

def set_max_processes(m):
    global _MAX_PROCESS
    _MAX_PROCESS = m

def stop():
    global _STOP
    _STOP = True

def launch():
    global _STOP
    _STOP = False
    for p in range(_MAX_PROCESS):
        thr = threading.Thread(target=_thread)
        thr.start()

def _thread():
    global _NB_EXECUTED
    while True:
        with _RLOCK:
            try:
                cmd = _QUEUE_COMMAND.get_nowait()
            except:
                cmd = None
        if cmd is not None:
            p = mp.Process(target=_process, args=(cmd,))
            p.start()
            p.join()
            _NB_EXECUTED += 1
            print(f"{_NB_EXECUTED} were executed")
        if _STOP:
            return
        time.sleep(0.01)

def _process(cmd):
    os.system(cmd)
