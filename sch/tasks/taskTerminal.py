import os
import json
from taskqueue import TaskQueue


file_path = "./tasks/taskqueue.json"

if os.path.exists(file_path):
        os.remove(file_path)
with open(file_path, "w") as file:
    pass

task_queue = TaskQueue(file_path)
while(1):
    command = input("#")
    if command == "add":
        task_json = input("task_json:")
        task_queue.add_task(task_json)
        print("successful")
    elif command == "queue":
        print("task queue:", task_queue.get_queue())
    elif command == "exit":
        break
    else:
        print("Invalid command.")