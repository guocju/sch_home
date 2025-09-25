import glob
import numpy as np
import time
import tvm
from tvm import relay
import onnx

file_paths = glob.glob("data/*")
inputs = [np.load(path) for path in file_paths]
# onnx_model = onnx.load("cagyolov7-tiny-s0_llvip_512x768.onnx")
# mod, params = relay.frontend.from_onnx(onnx_model)
# tvm_target = tvm.target.iluvatar(options="-libs=cudnn,cublas,ixinfer")
# with tvm.transform.PassContext(opt_level=2):
#    vm_exec = relay.vm.compile(mod, target=tvm_target, params=params)

start = time.time()
lib = tvm.runtime.load_module("device/GPU/GPU_yolo.so")
with open("device/GPU/GPU_yolo.bin", "rb") as f:
    code = f.read()
exe = tvm.runtime.vm.Executable.load_exec(code, lib)
the_vm = tvm.runtime.vm.VirtualMachine(exe, tvm.iluvatar())
outputs = []
for input in inputs:
    result = the_vm.invoke("main", tvm.nd.array(input))
    outputs.append(result[0].numpy())
    print("done")
print(len(outputs))
end = time.time()
total = end - start
fps = 1000/total
print(f"fps {fps}")


