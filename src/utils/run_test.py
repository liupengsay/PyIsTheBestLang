import os
import random

import unittest
from collections import defaultdict


class TestGeneral(unittest.TestCase):

    def test_add_solution_http(self):

        def process_file(file_path):
            nonlocal drop_dup
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = [ls.strip("\n") for ls in file.readlines()]
            lst = []
            dct = defaultdict(list)
            for line in lines:
                st = "（https://"
                try:
                    if st in line and "https://codeforces.com" in line:
                        idx = line.split(st)[0]
                        url = "https://" + line.split(st)[1].split("）")[0]
                        tag = line.split(st)[1].split("）")[1]
                        dct[idx.lower()] = ["        \"\"\"", f"        url: {url}", f"        tag: {tag}", "        \"\"\""]
                except:
                    print(file_path, line)

                st = "def cf_"
                if lst and st in lst[-1]:
                    try:
                        lst.extend(dct[lst[-1].split("(")[0].split("cf_")[1].split("_")[0]])
                    except:
                        print(file_path, lst[-1])

                lst.append(line)
            with open(file_path, "w", encoding="utf-8", errors="ignore") as file:
                file.write("\n".join(lst))
            return

        def process_directory(directory):
            nonlocal pre
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file == "problem.py":
                        file_path = os.path.join(root, file)
                        process_file(file_path)
                        pre += 1
            return

        current_path = os.getcwd()
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        grandparent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
        drop_dup = 0
        pre = 0
        process_directory(os.path.join(grandparent_path, "src"))
        print(f"total time cost：{drop_dup}")
        return

    @unittest.skip
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

    @unittest.skip
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

    @unittest.skip
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

    @unittest.skip
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

    @unittest.skip
    def test_drop_dup_problem(self):

        def process_file(file_path):
            nonlocal drop_dup
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = [ls.strip("\n") for ls in file.readlines()]
            lst = []
            for line in lines:
                st = "（https://"
                if st in line and line in pre:
                    drop_dup += 1
                    continue
                else:
                    lst.append(line)
                    pre.add(line)
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
        pre = set()
        drop_dup = 0
        process_directory(os.path.join(grandparent_path, "src"))
        print(f"total drop_dup：{drop_dup}")
        return

    @unittest.skip
    def test_remove_ref_and_problem(self):

        def process_file(file_path):
            nonlocal rem
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = [ls.strip("\n") for ls in file.readlines()]
            lst = []
            for line in lines:
                if "题目：" in line or "参考：" in line:
                    rem += 1
                    continue
                else:
                    lst.append(line)
                    pre.add(line)
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
        pre = set()
        rem = 0
        process_directory(os.path.join(grandparent_path, "src"))
        print(f"total rem：{rem}")
        return


if __name__ == '__main__':
    unittest.main()
