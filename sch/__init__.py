from .device.devicePool import cpu, gpu, npu, fpga
from multiprocessing.managers import BaseManager
from typing import Union, Callable, Any
import traceback
import tvm
from tvm.ir.module import IRModule
from pebble import ThreadPool
from concurrent.futures import TimeoutError
import threading
import time

lock = threading.Lock()
condition = threading.Condition(lock)
class MyManager(BaseManager): pass

MyManager.register('register_task')
MyManager.register('increase_task')
MyManager.register('decrease_task')
MyManager.register('get_strategy')

mgr = MyManager(address="/tmp/scheduler.sock", authkey=b'lemon')
mgr.connect()

str_to_dev = {"CPU": cpu,
              "GPU": gpu,
              "NPU": npu,
              "FPGA": fpga}

def _worker_call(function, index, func_args):
    result = function(*func_args)
    return index, result

class TaskService:
    def __init__(self, max_workers=8):
        self.task_dict = {} # {task_type: usr_dict}
        self.dev_state = {} # {device: is_free}
        self.inp_counter = {} # {task_type: counter}
        self.oup_counter = {} # {task_type: counter}
        self.task_strategy = {} # {task_type: strategy}
        self.total_time = 0
        self.batch_size = 20
        self.task_num = 0
        self.pool = ThreadPool(max_workers=max_workers)
    
    @staticmethod
    def load_lib(dev, executor_kind, so_path):
        device = str_to_dev[dev]
        return device.load_lib(executor_kind, so_path)
    
    def runTask(self, task_type:str, inputs:Any):
        batch_size = self.batch_size
        dev_dict = self.task_dict[task_type]
        
        with condition:
            if self.inp_counter[task_type] == 0:
                mgr.increase_task(task_type)
                strategy = mgr.get_strategy(task_type)
                self.task_strategy[task_type] = strategy.copy()
            self.inp_counter[task_type] += 1
            if self.inp_counter[task_type] % batch_size == 0:
                strategy = mgr.get_strategy(task_type)
                self.task_strategy[task_type] = strategy.copy()
            
        strategy = self.task_strategy[task_type]
        
        with condition:  # 自动 acquire + release
            free_dev = None
            while not free_dev:
                for dev in strategy:
                    if self.dev_state[dev]:
                        free_dev = dev
                        self.dev_state[dev] = 0
                        break
                if not free_dev:
                    condition.wait()
        executor_kind, exe = dev_dict[free_dev]
        device = str_to_dev[free_dev]
        result = device.compute(executor_kind, exe, inputs)
        with condition:
            self.dev_state[free_dev] = 1
            condition.notify_all()
        with condition:
            self.oup_counter[task_type] += 1
            if self.oup_counter[task_type] == self.task_num:
                mgr.decrease_task(task_type)
        return result
        

    def registerTask(self, task_type:str, devices:dict[str, float], IR: Union[IRModule, str], params = None):
        usr_dict = {}
        for dev, affinity in devices.items():
            if dev not in self.dev_state:
                self.dev_state[dev] = 1
            device = str_to_dev[dev]
            executor_kind, so_path = device.build(task_type, IR, params)
            mgr.register_task(dev, task_type, affinity, executor_kind, so_path)
            exe = TaskService.load_lib(dev, executor_kind, so_path)
            usr_dict[dev] = (executor_kind, exe)
        self.task_dict[task_type] = usr_dict
        self.inp_counter[task_type] = 0
        self.oup_counter[task_type] = 0
        

    def runTaskMultiThread(self,
                        function: Callable[..., Any],
                        Inputs: list[Any]):
        n = len(Inputs)
        self.task_num = n
        results: list[Any] = [None] * n
        exceptions: list[BaseException] = [None] * n

        future_to_idx = {}
        def worker_closure(idx, inp):
            # 只传入 idx 和 input
            func_args = (self,) + (tuple(inp) if isinstance(inp, (tuple, list)) else (inp,))
            return _worker_call(function, idx, func_args)
        for idx, inp in enumerate(Inputs):
            future = self.pool.schedule(worker_closure, args=(idx, inp))
            future_to_idx[future] = idx
        for future in future_to_idx:
            try:
                _idx, value = future.result()
                results[_idx] = value
            except TimeoutError:
                print(f"[Worker error] task {future_to_idx[future]} timeout")
            except Exception as exc:
                idx = future_to_idx[future]
                exceptions[idx] = exc
                print("[Worker error] exception received in parent:")
                traceback.print_exception(type(exc), exc, exc.__traceback__)

        return results

def connect():
    return TaskService()
