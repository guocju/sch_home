import os

from multiprocessing.managers import BaseManager

from schedule.scheduler import Scheduler
from device.devicePool import cpu, gpu, npu, fpga

sched = Scheduler()

def register_task(dev, task_type, affinity, executor, so_path):
    sched.register_task(dev, task_type, affinity, executor, so_path)

def increase_task(task_type:str):
    sched.increase_task(task_type)
    
def decrease_task(task_type:str):
    sched.decrease_task(task_type)

def get_strategy(task_type):
    return sched.best_strategy[task_type]

if __name__ == "__main__":
    gpu0 = gpu(0)
    cpu0 = cpu(0)
    sched.addDev(gpu0)
    sched.addDev(cpu0)
    sched.start_plot()
    sched.listen_command()
    class MyManager(BaseManager): pass      
    
    socket_file = "/tmp/scheduler.sock"
    if os.path.exists(socket_file):
        os.remove(socket_file)

    mgr = MyManager(address=socket_file, authkey=b'lemon')
    MyManager.register('register_task', callable=register_task)
    MyManager.register('increase_task', callable=increase_task)
    MyManager.register('decrease_task', callable=decrease_task)
    MyManager.register('get_strategy', callable=get_strategy)
    server = mgr.get_server()
    print(f"Scheduler RPC server listening on {socket_file}")
    server.serve_forever()
    
