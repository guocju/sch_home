from tasks.task import Task
import os
import fcntl

class TaskQueue:
    def __init__(self, tasks_file="./tasks/taskqueue.json"):
        self.tasks_file = tasks_file
        self.queue = self.load_tasks()
        self.max_tasks = 5
        
    def load_tasks(self):
        if not os.path.exists(self.tasks_file):
            return []
        with open(self.tasks_file, "r", encoding="utf-8") as f:
            try:
                return [Task.from_json(task_str) for task_str in f.readlines()]
            except ValueError:
                return []

    def save_tasks(self):
        with open(self.tasks_file, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            for task in self.queue:
                f.write(repr(task) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def add_task(self, json_str: str):
        self.queue = self.load_tasks()
        if len(self.queue) >= self.max_tasks:
            print(f"can't add new task. max = {self.max_tasks}")
            return
        new_task = Task.from_json(json_str)
        if any(task.name == new_task.name for task in self.queue):
            print(f"Task with name '{new_task.name}' already exists.")
            return
        self.queue.append(new_task)
        self.save_tasks()

    def complete_task(self, name: str):
        self.queue = self.load_tasks()
        self.queue = [task for task in self.queue if task.name != name]
        self.save_tasks()

    def get_queue(self):
        self.queue = self.load_tasks()
        return [repr(task) for task in self.queue]

if __name__ == "__main__":
    task_queue = TaskQueue()

    task1_json = '{"name": "Task1", "type": "yolo", "source_addr": "/path/to/source", "num": 10}'
    task_queue.add_task(task1_json)
    task2_json = '{"name": "Task2", "type": "yolo", "source_addr": "/path/to/source", "num": 20}'
    task_queue.add_task(task2_json)
    task3_json = '{"name": "Task3", "type": "BFS", "source_addr": "/path/to/source", "num": 30}'
    task_queue.add_task(task3_json)
    print("task queue:", task_queue.get_queue())

    task_queue.complete_task("Task1")
    print("task queue:", task_queue.get_queue())
    task_queue.complete_task("Task2")
    print("task queue:", task_queue.get_queue())
    