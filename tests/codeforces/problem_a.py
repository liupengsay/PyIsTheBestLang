import random
from sys import stdin
from typing import List

inf = 1 << 68


class FastIO:
    def __init__(self):
        self.random_seed = random.randint(0, 10 ** 9 + 7)
        return

    @staticmethod
    def read_int():
        return int(stdin.readline().rstrip())

    @staticmethod
    def read_float():
        return float(stdin.readline().rstrip())

    @staticmethod
    def read_list_ints():
        return list(map(int, stdin.readline().rstrip().split()))

    @staticmethod
    def read_list_ints_minus_one():
        return list(map(lambda x: int(x) - 1, stdin.readline().rstrip().split()))

    @staticmethod
    def read_str():
        return stdin.readline().rstrip()

    @staticmethod
    def read_list_strs():
        return stdin.readline().rstrip().split()

    @staticmethod
    def st(x):
        return print(x)

    @staticmethod
    def lst(x):
        return print(*x)

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def ceil(a, b):
        return a // b + int(a % b != 0)

    @staticmethod
    def accumulate(nums):
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        return pre




class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        tree = RangeChangeAddRangeMax(n)
        tree.build(nums)
        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, k = lst[1:]
                tree.range_change_add(x - 1, y - 1, tree.change_to_mask(k))
                for i in range(x - 1, y):
                    nums[i] = k
            elif lst[0] == 2:
                x, y, k = lst[1:]
                tree.range_change_add(x - 1, y - 1, tree.add_to_mask(k))
                for i in range(x - 1, y):
                    nums[i] += k
            else:
                x, y = lst[1:]
                ac.st(tree.range_max(x - 1, y - 1))
        return


Solution().main()
