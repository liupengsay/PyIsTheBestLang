import os
import random

import unittest


class TestGeneral(unittest.TestCase):

    def test_run_example(self):

        def run_example(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file == 'example.py':
                        file_path = os.path.join(root, file)
                        print(f"Running: {file_path}")
                        os.system(f"python {file_path}")
            return

        # executing all the example.py
        current_path = os.getcwd()
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        grandparent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
        run_example(os.path.join(grandparent_path, "src"))
        return

    def test_run_problem(self):

        def process_file(file_path):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()
            filtered_lines = ["https" + line.strip("\n").split("（https")[1].split("）")[0] for line in lines if
                              "（https" in line]
            tot.extend(filtered_lines)
            return

        def process_directory(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file == "problem.py":
                        file_path = os.path.join(root, file)
                        process_file(file_path)
            return

        # get all the problem.py and shuffle the list
        current_path = os.getcwd()
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        grandparent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
        tot = []
        process_directory(os.path.join(grandparent_path, "src"))
        random.shuffle(tot)
        with open(os.path.join(grandparent_path, "data/Problem.md"), 'w', encoding="utf-8", errors="ignore") as f:
            f.writelines("\n".join(tot))
        return

    def test_change_site(self):

        def process_file(file_path):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = [ls.strip("\n") for ls in file.readlines()]
            lst = []
            for line in lines:
                st = "AcWing"
                m = len(st)
                pre = (80 - m) // 2
                post = 80 - m - pre
                if f"={st}=" in line:
                    line = "=" * pre + st + "=" * post
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

        current_path = os.getcwd()
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        grandparent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
        process_directory(os.path.join(grandparent_path, "src"))
        return

    def test_change_cf_title(self):

        def process_file(file_path):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = [ls.strip("\n") for ls in file.readlines()]
            lst = []
            for line in lines:
                st = "（https://codeforces.com"
                if st in line:
                    line = line.split(st)
                    tmp = line[1].split("）")[0].split("/")
                    number = ""
                    for s in tmp[1:-1]:
                        if s.isnumeric():
                            number = s
                    number += tmp[-1]
                    line[0] = number
                    lst.append(st.join(line))
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

        current_path = os.getcwd()
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        grandparent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
        process_directory(os.path.join(grandparent_path, "src"))
        return


if __name__ == '__main__':
    unittest.main()
