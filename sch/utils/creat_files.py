import numpy as np
import os

def generate_and_save(output_dir: str, count: int = 100):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(count):
        # 随机生成 1×3×224×224 的 float32 矩阵
        arr = np.random.rand(1, 3, 512, 768).astype('float32')
        # 构造文件名，例如 "0.npy", "1.npy", ..., "99.npy"
        filename = os.path.join(output_dir, f"{i}.npy")
        # 保存为 .npy 文件
        np.save(filename, arr)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    generate_and_save(output_dir="../data", count=1000)
