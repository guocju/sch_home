from .ability import Ability
import threading
import os
import time
import tvm
from tvm import relay
from tvm.ir.module import IRModule
import onnx

to_tvm_device = {"CPU":tvm.cpu(),
                "GPU":tvm.iluvatar(),
                "NPU":"npu",
                "FPGA":"fpga"}

to_tvm_target = {"CPU":"llvm",
               "GPU":tvm.target.iluvatar(options="-libs=cudnn,cublas,ixinfer"),
               "NPU":"npu",
               "FPGA":"fpga"}

lock = threading.Lock()

class Device:
    input_pointer = {}# {task_type: pointer}
    output_pointer = {}
    task_inputs = {} #{task_type: list(inputs)}
    task_outputs = {}
    need_schedule = 0
    CallBackFunction = None
    
    def __init__(self, id:int):
        self.id = id
        self.ComputePower = 0
        self.is_free = 1
        self.lib_loaded = 0
        self.ability = {}
        self.task_type = []
        self.equivalent_power = 0
        self.task_counter = 0
        self.DeviceType = "base device"
        self.exe = {}
        self.task_fps = []# [[start_time, fps]]
    
    def add_ability(self, task_type, affinity, ir_type, so_path):
        ability = Ability(task_type, affinity, so_path, ir_type)
        self.ability[task_type] = ability
    
    def start(self):
        while True:
            time.sleep(0.05)
            while self.need_schedule:
                self.is_free = 1
                time.sleep(0.05)
            while self.lib_loaded:
                self.is_free = 1
                if self.need_schedule:
                    break
                lock.acquire()
                if not self.lib_loaded:
                    lock.release()
                    break
                task_id = self.task_counter
                task_type = self.task_type[task_id]
                task_num = len(self.task_inputs[task_type])
                pointer = self.input_pointer[task_type]
                if pointer == task_num:
                    time.sleep(0.1)
                    task_num = len(self.task_inputs[task_type])
                    pointer = self.input_pointer[task_type]
                    if pointer == task_num:
                        print(f"Algorithm {task_type} is all done, task num {task_num}.")
                        while(not self.output_pointer[task_type] == task_num):
                            time.sleep(0.05)
                        self.task_outputs.pop(task_type)
                        self.task_inputs.pop(task_type)
                        self.input_pointer.pop(task_type)
                        self.output_pointer.pop(task_type)
                        self.CallBackFunction("Algorithm_done")
                        lock.release()
                        break
                
                ir_type = self.ability[task_type].ir_type
                task_input = self.task_inputs[task_type][pointer]
                pointer += 1
                self.input_pointer[task_type] = pointer
                lock.release()
                self.is_free = 0
                self.compute(task_type, task_input, ir_type, pointer - 1)
                print(f"{pointer} done")
                end_time = time.time()
                batch_time = end_time - self.task_fps[task_id][0]
                self.task_fps[task_id][0] = end_time
                self.task_fps[task_id][1] = 1 / batch_time
                task_id = (task_id + 1)%len(self.task_type)
                self.task_counter = task_id
        
            
    def run_task(self, task_type:str, inputs):
        if task_type in self.task_inputs:
            self.task_outputs[task_type].append(None)
            self.task_inputs[task_type].append(inputs)
            id = len(self.task_inputs[task_type]) - 1
            return id
        else:
            lock.acquire()
            self.task_outputs[task_type] = [None]
            self.task_inputs[task_type] = [inputs]
            self.input_pointer[task_type] = 0
            self.output_pointer[task_type] = 0
            self.CallBackFunction("new_task_type")
            id = len(self.task_inputs[task_type]) - 1
            lock.release()
            return id
    
    def get_output(self, task_type, id):
        while self.task_outputs[task_type][id] is None:
            time.sleep(0.1)
        output = self.task_outputs[task_type][id]
        self.output_pointer[task_type] += 1
        return output
            
    def __repr__(self):
        return self.DeviceType+"_"+str(self.id)
        
class cpu(Device):
    def __init__(self, id: int = 0):
        super().__init__(id)
        self.DeviceType = "CPU"
        self.ComputePower = 40 # 算力
        
    def build(task_type:str, IR, params = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(base_dir, "CPU", f"CPU_{task_type}.so")
        code_path = os.path.join(base_dir, "CPU", f"CPU_{task_type}.bin")
        if not os.path.exists(so_path):
            if isinstance(IR, IRModule):
                mod = IR
            elif isinstance(IR, str):
                onnx_model = onnx.load(IR)
                mod, params = relay.frontend.from_onnx(onnx_model)
            tvm_target = to_tvm_target["CPU"]
            with tvm.transform.PassContext(opt_level=2):
                vm_exec = relay.vm.compile(mod, target=tvm_target, params=params)
            print("build complete")
            code, lib = vm_exec.save()
            with open(code_path, "wb") as f:
                f.write(code)
            lib.export_library(so_path)
            print(f"saved to {so_path}")
        return "relayVM", so_path
        
    def load_lib(executor_kind, so_path):
        if executor_kind == "relayVM":
            path, ext = os.path.splitext(so_path)
            code_path = path + ".bin"
            lib = tvm.runtime.load_module(so_path)
            with open(code_path, "rb") as f:
                code = f.read()
            exe = tvm.runtime.vm.Executable.load_exec(code, lib)
            the_vm = tvm.runtime.vm.VirtualMachine(exe, to_tvm_device["CPU"])
            return the_vm
                    
    def compute(executor_kind, exe, input):
        result = None
        if executor_kind == "relayVM":
            result = exe.invoke("main", tvm.nd.array(input))
        result = result[0].numpy()
        return result
        
class gpu(Device):
    def __init__(self, id: int = 0):
        super().__init__(id)
        self.DeviceType = "GPU"
        self.ComputePower = 500 # 算力
        
    def build(task_type:str, IR, params = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(base_dir, "GPU", f"GPU_{task_type}.so")
        code_path = os.path.join(base_dir, "GPU", f"GPU_{task_type}.bin")
        if not os.path.exists(so_path):
            if isinstance(IR, IRModule):
                mod = IR
            elif isinstance(IR, str):
                onnx_model = onnx.load(IR)
                mod, params = relay.frontend.from_onnx(onnx_model)
            tvm_target = to_tvm_target["GPU"]
            with tvm.transform.PassContext(opt_level=2):
                vm_exec = relay.vm.compile(mod, target=tvm_target, params=params)
            print("build complete")
            code, lib = vm_exec.save()
            with open(code_path, "wb") as f:
                f.write(code)
            lib.export_library(so_path)
            print(f"saved to {so_path}")
        return "relayVM", so_path
        
    def load_lib(executor_kind, so_path):
        if executor_kind == "relayVM":
            path, ext = os.path.splitext(so_path)
            code_path = path + ".bin"
            lib = tvm.runtime.load_module(so_path)
            with open(code_path, "rb") as f:
                code = f.read()
            exe = tvm.runtime.vm.Executable.load_exec(code, lib)
            the_vm = tvm.runtime.vm.VirtualMachine(exe, to_tvm_device["GPU"])
            return the_vm
                    
    def compute(executor_kind, exe, input):
        result = None
        if executor_kind == "relayVM":
            result = exe.invoke("main", tvm.nd.array(input))
        result = result[0].numpy()
        return result
               

class npu(Device):
    def __init__(self, id: int = 0):
        super().__init__(id)
        self.DeviceType = "NPU"
        self.ComputePower = 200 # 算力
              

class fpga(Device):
    def __init__(self, id: int = 0):
        super().__init__(id)
        self.DeviceType = "FPGA"
        self.ComputePower = 100 # 算力
                
