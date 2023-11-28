import os
import time


def run_example_files(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file == 'example.py':
                file_path = os.path.join(root, file)
                print(f"Running: {file_path}")
                os.system(f"python {file_path}")  # 在命令行中运行 example.py 文件
    return


current_path = os.getcwd()  # 获取当前路径
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))  # 获取父级路径
grandparent_path = os.path.abspath(os.path.join(parent_path, os.pardir))  # 获取上上级路径
# 示例用法
s = time.time()
run_example_files(os.path.join(grandparent_path, "src"))
print("总耗时: ", int(time.time() - s), "s")
