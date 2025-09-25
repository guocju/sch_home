import sch
import glob
import numpy as np
import time

def app(svc, input):
    # preprocess code
    res = svc.runTask("yolo", input)
    # postprocess code
    return res

if __name__ == "__main__":
    dev_dict = {"CPU": 0.9, "GPU": 0.7}
    svc = sch.connect()
    svc.registerTask("yolo", dev_dict, "cagyolov7-tiny-s0_llvip_512x768.onnx")
    file_paths = glob.glob("data/*")
    inputs = [np.load(path) for path in file_paths]
    
    start = time.time()
    outs = svc.runTaskMultiThread(app, inputs)
    end = time.time()
    total = end - start
    print(1000/total)
    print(len(outs))
