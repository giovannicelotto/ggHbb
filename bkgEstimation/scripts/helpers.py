import numpy as np
import time
import fcntl

def acquire_lock(lockPath):
    while True:
        try:
            lock_fd = open(lockPath, "w")
            fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_fd
        except IOError:
            time.sleep(1)

def release_lock(lock_fd):
    fcntl.lockf(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()