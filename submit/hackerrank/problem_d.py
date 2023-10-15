import random
import sys
import bisect
import decimal
import heapq
from types import GeneratorType
from math import inf

from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heappushpop
from functools import cmp_to_key
from collections import defaultdict, Counter, deque
import math
from functools import lru_cache
from heapq import nlargest
from functools import reduce
from decimal import Decimal

from itertools import combinations, permutations
from operator import xor, add
from operator import mul
from typing import List, Callable, Dict, Set, Tuple, DefaultDict
from heapq import heappush, heappop, heapify


RANDOM = random.randint(0, 10**9 + 7)


class Wrapper(int):
    def __init__(self, x):
        int.__init__(x)

    def __hash__(self):
        return super(Wrapper, self).__hash__() ^ RANDOM


class FastIO:
    def __init__(self):
        self.random_seed = random.randint(0, 10 ** 9 + 7)
        return

    @staticmethod
    def read_int():
        return int(sys.stdin.readline().strip())

    @staticmethod
    def read_float():
        return float(sys.stdin.readline().strip())

    @staticmethod
    def read_list_ints():
        return list(map(int, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_floats():
        return list(map(float, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_ints_minus_one():
        return list(map(lambda x: int(x) - 1, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_str():
        return sys.stdin.readline().strip()

    @staticmethod
    def read_list_strs():
        return sys.stdin.readline().strip().split()

    @staticmethod
    def read_list_str():
        return list(sys.stdin.readline().strip())

    @staticmethod
    def st(x):
        return print(x)

    @staticmethod
    def lst(x):
        return print(*x)

    @staticmethod
    def round_5(f):
        res = int(f)
        if f - res >= 0.5:
            res += 1
        return res

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def ceil(a, b):
        return a // b + int(a % b != 0)

    def hash_num(self, x):
        return x ^ self.random_seed

    @staticmethod
    def accumulate(nums):
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        return pre

    def inter_ask(self, lst):
        self.st(lst)
        sys.stdout.flush()
        res = self.read_int()
        return res

    def inter_out(self, lst):
        self.st(lst)
        sys.stdout.flush()
        return


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):

        return


Solution().main()
