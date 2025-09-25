## 调度器框架使用说明

调度器里面使用到TVM，事先确保TVM已经安装好。

### 第一步 设置环境变量

将调度器模块sch导入到python path里面。

```bash
export PYTHONPATH=/path/to/sch_home:${PYTHONPATH}
```

### 第二步 启动调度器进程

运行sch_home/sch/main.py，启动调度器服务进程。

出现以下输出表示启动成功，调度器进程常驻

```bash
INFO:     Started server process [8369]
INFO:     Waiting for application startup.
INFO:schedule.plot:[task_plot] Server starting on 127.0.0.1:2000
INFO:     Application startup complete.
INFO:schedule.plot:Server started at http://127.0.0.1:2000
INFO:     Started server process [8369]
INFO:     Waiting for application startup.
INFO:schedule.plot:[task_plot] Server starting on 127.0.0.1:1900
INFO:     Application startup complete.
INFO:schedule.plot:Server started at http://127.0.0.1:1900
INFO:     Uvicorn running on http://127.0.0.1:1900 (Press CTRL+C to quit)
#Scheduler RPC server listening on /tmp/scheduler.sock
INFO:     Uvicorn running on http://127.0.0.1:2000 (Press CTRL+C to quit)
```

### 第三步 运行用户脚本

用户进程和调度器进程使用进程间通信，因此，用户可以在本机的任意位置运行任务脚本。

其中一个示例脚本是test_yolo.py

使用GPU进行yolo v7 tiny的ONNX网络推理，1000个npy输入大约可以跑到160帧，首次运行会进行tvm编译。

用户脚本运行完成后打印出帧数和任务总数后自然退出。



## 调度器设备添加方法

如果需要向框架中添加新的device，需要在device/devicePool.py里面添加设备

同时也需要增加main.py和_init__.py的部分代码。

### 第一步 定义设备类和类方法

每一种设备需要注册一个class，继承自Device基类，然后为每一个设备实现build, load_lib, compute三种函数。

```python
def build(task_type:str, IR, params = None):
    ...
	return executor_kind, so_path
def load_lib(executor_kind, so_path):
    ...
	return executor
def compute(executor_kind, exe, input):
    ...
    return result
```

### 第二步 增加设备到调度器

可以在main.py添加相应设备代码：

```python
if __name__ == "__main__":
    gpu0 = gpu(0)
    cpu0 = cpu(0)
    sched.addDev(gpu0)
    sched.addDev(cpu0)
```

打开_init__.py增加映射：

```python
str_to_dev = {"CPU": cpu,
              "GPU": gpu,
              "NPU": npu,
              "FPGA": fpga}
```

