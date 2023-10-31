import os


def run_example_files(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file == 'example.py':
                file_path = os.path.join(root, file)
                print(f"Running: {file_path}")
                os.system(f"python3 {file_path}")  # 在命令行中运行 example.py 文件
    return


# 示例用法
directory = 'D:\\AlgorithmPY\\src'
run_example_files(directory)
