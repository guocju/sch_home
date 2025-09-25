from itertools import chain, combinations
from device.devicePool import Device
from tasks.task import Task
import threading
import time
from schedule.plot import device_port, TaskPlotServer

lock = threading.Lock()

class Scheduler:
    devs = []
    is_dynamic = 0
    task_counter = {}# {task_type: num}
    best_strategy = {} # {task_type: list[dev.DeviceType]}
    
    def addDev(self, dev:Device):
        self.devs.append(dev)
        
    def register_task(self, dev:str, task_type:str, affinity:float, ir_type:str, so_path:str):
        for device in self.devs:
            if device.DeviceType == dev:
                device.add_ability(task_type, affinity, ir_type, so_path)
    
    def increase_task(self, task_type:str):
        if task_type in self.task_counter:
            self.task_counter[task_type] += 1
        else:
            self.task_counter[task_type] = 1
            self.on_event("new_task_type")
        
        
    def decrease_task(self, task_type:str):
        self.task_counter[task_type] -= 1
        if self.task_counter[task_type] == 0:
            self.on_event("Algorithm_done")
            self.task_counter.pop(task_type)
    
    def switch_mode(self):
        mode = self.is_dynamic
        mode = 1 - mode
        self.is_dynamic = mode
        self.on_event("switch")
    
    def on_event(self, event_kind):
        print(f"[Scheduler] 收到 {event_kind}")
        task_kinds = list(self.task_counter.keys())
                
        if event_kind == "new_task_type":
            self.find_best_strategy(task_kinds)
        elif event_kind == "Algorithm_done":
            self.find_best_strategy(task_kinds)
        elif event_kind == "switch":
            self.find_best_strategy(task_kinds)
                
    def find_dynamic_strategy(self, task_kinds:list, devices:list):
        max_power = 0
        device_combinations = [()]
        for num_devices in range(1, len(devices) + 1):
            device_combinations.extend(combinations(devices, num_devices))
        stack = [(task_kinds, [])]

        while stack:
            remaining_tasks, current_assignment = stack.pop()
            if not remaining_tasks:
                if Scheduler.is_rational(current_assignment):
                    for dev in devices:
                        dev.task_type = []
                    for task, assigned_devices in current_assignment:
                        for device in assigned_devices:
                            device.task_type.append(task)
                    compute_power = 0
                    for dev in devices:
                        equivalent_power = 0
                        if not dev.task_type:
                            continue
                        for task in dev.task_type:
                            equivalent_power += dev.ComputePower*dev.ability[task].affinity
                        dev.equivalent_power = equivalent_power/len(dev.task_type)
                        compute_power += dev.equivalent_power
                    if compute_power > max_power:
                        max_power = compute_power
                        best_strategy = current_assignment
                continue
            
            current_task = remaining_tasks[0]
            rest_tasks = remaining_tasks[1:]
            for comb in device_combinations:
                new_assignment = current_assignment + [(current_task, comb)]
                stack.append((rest_tasks, new_assignment))
        return best_strategy
    
    def find_static_strategy(self, task_kinds:list, devices:list):
        best_strategy = []
        for dev in devices:
            dev.task_type = []
        for task_str in task_kinds:
            max_power = 0
            best_device = None
            for dev in devices:
                if task_str not in dev.ability:
                    continue
                device_power = dev.ComputePower*dev.ability[task_str].affinity
                if device_power > max_power:
                    max_power = device_power
                    best_device = dev
            best_strategy.append((task_str, [best_device]))
        return best_strategy
        
    def find_best_strategy(self, task_kinds:list):
        devices = self.devs
        best_strategy = None
        if not task_kinds:
            for dev in devices:
                dev.task_type = []
            return
        if self.is_dynamic:
            best_strategy = self.find_dynamic_strategy(task_kinds, devices)
        else:
            best_strategy = self.find_static_strategy(task_kinds, devices)
        
        for dev in devices:
            dev.task_type = []
            dev.task_fps = []
        new_best_strategy = {}
        for task, assigned_devices in best_strategy:
            new_best_strategy[task] = []
            for device in assigned_devices:
                new_best_strategy[task].append(device.DeviceType)
                device.task_type.append(task)
                start_time = time.time()
                device.task_fps.append([start_time, 0])
        self.best_strategy = new_best_strategy
        for dev in devices:
            equivalent_power = 0
            if not dev.task_type:
                dev.equivalent_power = 0
                continue
            for task in dev.task_type:
                equivalent_power += dev.ComputePower*dev.ability[task].affinity
            dev.equivalent_power = equivalent_power/len(dev.task_type)
        
    def start_plot(self):
        dev_plots = {} # dev_type: plotter
        dev_job = {} # task_type: fps
        dev_fps = {} # dev_type: dev_job
        for dev in self.devs:
            dev_type = dev.DeviceType
            if dev_type not in dev_plots:
                device_plot = TaskPlotServer(host="127.0.0.1", port=device_port[dev_type])
                dev_plots[dev_type] = device_plot
                dev_fps[dev_type] = {}
                device_plot.start(background=True)
        
        def keep_plot():
            begin_time = time.time()
            while(1):
                dev_new_jobs = {} # dev_type: set[job_kinds]
                for dev_type in dev_fps:
                    dev_job = dev_fps[dev_type]
                    dev_job = {key: 0 for key in dev_job}
                    dev_fps[dev_type] = dev_job
                for dev in self.devs:
                    dev_type = dev.DeviceType
                    if dev_type not in dev_new_jobs:
                        dev_new_jobs[dev_type] = set()
                    for index, task_type in enumerate(dev.task_type):
                        if task_type not in dev_fps[dev_type]:
                            dev_plots[dev_type].add_task(task_type)
                            dev_fps[dev_type][task_type] = 0
                        dev_new_jobs[dev_type].add(task_type)
                        fps = dev.task_fps[index][1]
                        dev_fps[dev_type][task_type] += fps
                for dev_type, plotter in dev_plots.items():
                    for dev_job in list(dev_fps[dev_type]):
                        if dev_job not in dev_new_jobs[dev_type]:
                            dev_fps[dev_type].pop(dev_job)
                            plotter.remove_task(dev_job)
                    for job_type, fps in dev_fps[dev_type].items():
                        cur_time = time.time() - begin_time
                        plotter.push_value(job_type, cur_time, fps)
                time.sleep(0.1)

        t = threading.Thread(target=keep_plot)
        t.start()
        
    def listen_command(self):
        def keep_listen():
            while(1):
                command = input("#")
                if command == "switch":
                    if self.is_dynamic:
                        print("use static schedule")
                    else:
                        print("use dynamic schedule")
                    self.switch_mode()
                elif command == "exit":
                    break
                else:
                    print("Invalid command.")
                    
        t = threading.Thread(target=keep_listen)
        t.start()
    
    @staticmethod
    def is_rational(strategy):
        for task, assigned_devices in strategy:
            for device in assigned_devices:
                ability = list(device.ability.keys())
                if task not in ability:
                    return False
        return True