import onnx
import tvm
import numpy as np
from tvm import relay
import os
from tvm.contrib import graph_executor

# -----------------------------
# 1. 加载 ONNX 模型
# -----------------------------
onnx_model_path = "/home/xjtu3/proj/sch/cagyolov7-tiny-s0_llvip_512x768.onnx"  # 替换成你的模型文件
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
target = tvm.target.Target(target="iluvatar", host="llvm")  # 可以是 llvm / llvm -mcpu=core-avx2 / cuda 等
with tvm.transform.PassContext(opt_level=3):
     vm_exec = relay.vm.compile(mod, target=target, params=params)
print("build complete")
dev = tvm.iluvatar()
vm = tvm.runtime.vm.VirtualMachine(vm_exec, dev)
data = np.random.rand(1, 3, 512, 768).astype("float32")

vector_input = tvm.nd.array(data, dev)
result = vm.invoke("main", vector_input)
print(result.shape)

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
module = graph_executor.GraphModule(lib["default"](dev))
data = np.random.rand(1, 3, 512, 768).astype("float32")


module.set_input(0, data)
module.run()
out = module.get_output(0).numpy()
