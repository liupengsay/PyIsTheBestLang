import os
import random
import time


def process_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()
    filtered_lines = [line.strip("\n") for line in lines if "https" in line and "（https" not in line]
    tot.extend(filtered_lines)
    return


def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "problem.py":
                file_path = os.path.join(root, file)
                process_file(file_path)
    return


current_path = os.getcwd()  # 获取当前路径
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))  # 获取父级路径
grandparent_path = os.path.abspath(os.path.join(parent_path, os.pardir))  # 获取上上级路径
# 示例用法
s = time.time()
tot = []
process_directory(os.path.join(grandparent_path, "src"))
random.shuffle(tot)
with open(os.path.join(grandparent_path, "data/problem.txt"), 'w', encoding="utf-8", errors="ignore") as f:
    f.writelines("\n".join(tot))

print("总耗时: ", int(time.time() - s), "s")
