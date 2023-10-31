import os


def count_lines(file_path):
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()
        code_lines = 0
        for line in lines:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            code_lines += 1
        return code_lines


def count_lines_in_directory(directory):
    total_lines = 0
    files_counted = 0
    items = os.listdir(directory)
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            if item.endswith('.py'):
                lines = count_lines(item_path)
                print(f"{item_path}: {lines} lines")
                total_lines += lines
                files_counted += 1
        elif os.path.isdir(item_path):
            subdir_total_lines, subdir_files_counted = count_lines_in_directory(item_path)
            total_lines += subdir_total_lines
            files_counted += subdir_files_counted
    print(f"Total files counted: {files_counted}")
    print(f"Total lines of code: {total_lines}")
    return total_lines, files_counted


count_lines_in_directory('D:\\AlgorithmPY\\src')
