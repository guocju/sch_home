import json

class Task:
    VALID_TYPES = {"yolo", "BFS", "test"}

    def __init__(self, name: str, task_type: str, source_addr: str):
        if task_type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of the following: {', '.join(self.VALID_TYPES)}")
        
        self.name = name
        self.type = task_type
        self.source_addr = source_addr

    def __repr__(self):
        return json.dumps({
            "name": self.name,
            "type": self.type,
            "source_addr": self.source_addr,
        }, ensure_ascii=False)  # Output as JSON string

    @classmethod
    def from_json(cls, json_str: str):  # eg: {"name": "yolo_1", "type": "yolo", "source_addr": "/home/guocj/hpuPerform", "num": 520}
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        return cls(
            name=data.get("name"),
            task_type=data.get("type"),
            source_addr=data.get("source_addr")
        )

if __name__ == "__main__":
    task1 = Task("yolo_1", "yolo", "/home/guocj/hpuPerform")
    task1_str = repr(task1)
    print(task1_str)
    task2 = Task.from_json(task1_str)
