import os

def get_files_in_current_directory():
    # 获取当前路径
    current_directory = os.getcwd()

    # 获取当前路径下的文件列表
    files = os.listdir(current_directory)

    # 过滤掉文件夹，只保留文件
    files = [file for file in files if os.path.isfile(os.path.join(current_directory, file))]

    return files


import shutil

def copy_folder(source_folder, destination_folder):
    try:
        # 使用 shutil.copytree() 函数复制整个文件夹
        shutil.copytree(source_folder, destination_folder)
        print("文件夹复制成功")
    except shutil.Error as e:
        print(f"文件夹复制失败: {e}")
    except OSError as e:
        print(f"文件夹复制失败: {e}")

# 设置源文件夹路径和目标文件夹路径


# 调用函数获取当前路径下的文件列表
files = get_files_in_current_directory()

def move_file(source_file, destination_file):
    try:
        # 使用 shutil.move() 函数移动文件
        shutil.move(source_file, destination_file)
        print("文件移动成功")
    except shutil.Error as e:
        print(f"文件移动失败: {e}")
    except OSError as e:
        print(f"文件移动失败: {e}")


# 打印文件列表
for file in files:
    if ".py" in file and file != "temp.py":
        source_folder = "D:\\BackUP\\temp"
        destination_folder = file.replace(".py", "")
        # 调用函数复制文件夹
        copy_folder(source_folder, destination_folder)

        # 设置源文件路径和目标文件路径
        source_file = file
        destination_file = f'{destination_folder}/template.py'
        # 调用函数移动文件
        move_file(source_file, destination_file)
        print(file)