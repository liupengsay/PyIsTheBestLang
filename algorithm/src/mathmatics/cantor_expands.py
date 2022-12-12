"""
算法：康托展开
功能：康托展开可以用来求一个 1~n的任意排列的排名
题目：P5367 【模板】康托展开（https://www.luogu.com.cn/problem/P5367#submit）
参考：OI WiKi（https://oi-wiki.org/math/combinatorics/cantor/）
"""

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy


class CantorExpands:
    def __init__(self):
        return

    @lru_cache(None)
    def dfs(self, x):
        if x <= 1:
            return 1
        res = x * self.dfs(x - 1)
        return res

    def array_to_rank(self, nums):
        lens = len(nums)
        out = 0
        for i in range(lens):
            res = 0
            fact = self.dfs((lens - i - 1))
            for j in range(i + 1, lens):
                if nums[j] < nums[i]:
                    res += 1
            out += res * fact
        return out + 1

    def rank_to_array(self, n, k):
        nums = []
        for i in range(1, n + 1):
            nums.append(i)

        k = k - 1
        out = []
        while nums:
            p = k // self.dfs(len(nums) - 1)
            out.append(nums[p])
            k = k - p * self.dfs(len(nums) - 1)
            nums.pop(p)
        return out


class TestGeneral(unittest.TestCase):

    def test_cantor_expands(self):
        ce = CantorExpands()
        rk = 1
        for item in permutations(list(range(1, 8)), 7):
            lst = list(item)
            assert ce.rank_to_array(7, rk) == lst
            assert ce.array_to_rank(lst) == rk
            rk += 1
        return


if __name__ == '__main__':
    unittest.main()
