import os
import onnx
from tvm.ir.module import IRModule
from typing import Union, Callable, Any
from multiprocessing.managers import BaseManager
from concurrent.futures import ProcessPoolExecutor, as_completed

class Scheduler:
    def __init__(self):
        self.count = 0

    def schedule(self, task):
        self.count += 1
        print(f"[Scheduler] scheduling task: {task} (count={self.count})")
        return f"scheduled:{task}"
        
    def register_task(self, task_type:str, device:str, so_path:str):
        print("register_task")
        
    def run_task(self, task_type:str, input):
        print("run task")
        return 0
        
    def get_output(self, id:int):
        print("get output")
        return 0
        

# 单例 scheduler
scheduler = Scheduler()

def registerTask(task_type:str, devices:list, IR: Union[IRModule, str]):
    # relax mod
    if isinstance(IR, IRModule):
        os.environ["TVM_HOME"] = "/home/guocj/env/tvm"
        from tvm import relax
        
        for dev in devices:
            so_path = f"./device/{dev}/{dev}_{task_type}.so"
            ex = relax.build(IR, target=dev)
            ex.export_library(so_path)
            scheduler.register_task(task_type, dev, so_path)
            
    elif isinstance(IR, str):
        os.environ["TVM_HOME"] = "/home/guocj/env/tvm2"
        import tvm
        from tvm import relay
        
        for dev in devices:
            so_path = f"./device/{dev}/{dev}_{task_type}.so"
            onnx_model = onnx.load(IR)
            mod, params = relay.frontend.from_onnx(onnx_model)
            with tvm.transform.PassContext(opt_level=0):
                lib = relay.build(mod, target=dev, params=params)
            lib.export_library(so_path)
            scheduler.register_task(task_type, dev, so_path)
            
def runTask(task_type:str, inputs):
    id = scheduler.run_task(task_type, inputs)
    return scheduler.get_output(id)

def runTaskMultiProcess(function: Callable[..., Any],
                        Inputs: list[Any],
                        BatchSize: int = 10
                        ):
    if BatchSize <= 0:
        BatchSize = 1

    n = len(Inputs)
    results: list[Any] = [None] * n
    exceptions: list[BaseException] = [None] * n

    with ProcessPoolExecutor(max_workers=BatchSize) as exe:
        future_to_index = {}
        for idx, inp in enumerate(Inputs):
            args = tuple(inp) if isinstance(inp, (tuple, list)) else (inp,)
            fut = exe.submit(function, *args)
            future_to_index[fut] = idx

        for fut in as_completed(future_to_index):
            idx = future_to_index[fut]
            try:
                results[idx] = fut.result()
            except BaseException as e:
                exceptions[idx] = e

    return results
    
if __name__ == "__main__":
    class MyManager(BaseManager): pass      
    MyManager.register('registerTask', callable=registerTask)
    MyManager.register('runTask', callable=runTask)
    mgr = MyManager(address=('0.0.0.0', 50001), authkey=b'lemon')
    server = mgr.get_server()
    print("Scheduler RPC server listening on port 50001")
    server.serve_forever()
            