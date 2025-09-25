import onnx
import tvm
import numpy as np
from tvm import relay
import os

# -----------------------------
# 1. 加载 ONNX 模型
# -----------------------------
onnx_model_path = "/home/guocj/sch/cagyolov7-tiny-s0_llvip_512x768.onnx"  # 替换成你的模型文件
onnx_model = onnx.load(onnx_model_path)

# -----------------------------
# 2. 转成 Relay IR
# -----------------------------
input_name = "input"
shape_dict = {input_name: (1, 3, 512, 768)}
dtype_dict = {input_name: "float32"}

mod, params = relay.frontend.from_onnx(onnx_model)

# -----------------------------
# 3. 编译 Relay IR
# -----------------------------
target = tvm.target.Target("llvm")  # 可以是 llvm / llvm -mcpu=core-avx2 / cuda 等
with tvm.transform.PassContext(opt_level=3):
     vm_exec = relay.vm.compile(mod, target="llvm", params=params)
     
code, lib = vm_exec.save()
with open("code.bin", "wb") as f:
    f.write(code)
print("build complete")
dev = tvm.cpu()
vm = tvm.runtime.vm.VirtualMachine(vm_exec, dev)
data = np.random.rand(1, 3, 512, 768).astype("float32")

vector_input = tvm.nd.array(data, dev)
result = vm.invoke("main", vector_input)
print(result[0].numpy())

print("run complete")
# -----------------------------
# 4. 导出 .so 文件
# -----------------------------
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)
lib_path = os.path.join(output_dir, "model.so")

lib.export_library(lib_path)
print(f"Shared library saved to: {lib_path}")
lib = tvm.runtime.load_module(lib_path)
with open("code.bin", "rb") as f:
    code = f.read()
exe = tvm.runtime.vm.Executable.load_exec(code, lib)

# Benchmark the module.
the_vm = tvm.runtime.vm.VirtualMachine(exe, dev)
data = np.random.rand(1, 3, 512, 768).astype("float32")


vector_input = tvm.nd.array(data, dev)
result = the_vm.invoke("main", vector_input)
print(result[0].numpy())
