import numpy
from multiprocessing.managers import BaseManager

class MyManager(BaseManager): pass

MyManager.register('runTask')

def preprocess(input):
    print("preprocess")
    return input

def worker(input):
    gv1 = preprocess(input)
    mgr = MyManager(address=('127.0.0.1', 50001), authkey=b'lemon')
    mgr.connect()

    gv2 = mgr.runTask("yolo", gv1)
    return gv2