import os
import time


def process_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = [ls.strip("\n") for ls in file.readlines()]
    lst = []
    for line in lines:
        spl = "https://www.acwing.com"
        if spl in line:
            tmp = line.split(spl)
            tmp[0] = tmp[0].split(".")[0] + "（"
            line = spl.join(tmp)
            lst.append(line)
        else:
            lst.append(line)
    with open(file_path, "w", encoding="utf-8", errors="ignore") as file:
        file.write("\n".join(lst))
    return


def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "problem.py":
                file_path = os.path.join(root, file)
                process_file(file_path)
    return


# 示例用法
current_path = os.getcwd()  # 获取当前路径
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))  # 获取父级路径
grandparent_path = os.path.abspath(os.path.join(parent_path, os.pardir))  # 获取上上级路径
# 示例用法
s = time.time()
tot = []
process_directory(os.path.join(grandparent_path, "src"))
print("总耗时: ", int(time.time() - s), "s")
