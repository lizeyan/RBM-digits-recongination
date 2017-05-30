import time
from datetime import datetime
import numba as nb


__last_tic = None


def tic():
    global __last_tic
    __last_tic = time.time()


def toc():
    global __last_tic
    print("elapsed time: %f s" % (time.time() - __last_tic))
    __last_tic = None


def log(*args, **kwargs):
    print("[%s]" % str(datetime.now())[:-7], *args, **kwargs)


if __name__ == '__main__':
    tic()
    for i in range(10000):
        log("fuck")
    toc()

