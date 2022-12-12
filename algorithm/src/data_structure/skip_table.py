"""
算法：ST（跳表）
功能：计算静态区间内的最大值、最大公约数
题目：P3865 【模板】ST 表（https://www.luogu.com.cn/problem/P3865）
参考：OI WiKi（xx）
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


class SkipTable:
    def __init__(self, n, lst):
        self.n = n
        self.lst = lst
        self.f = [[0] * 18 for _ in range(self.n+1)]
        self.gen_skip_table()
        return

    def gen_skip_table(self):
        for i in range(1, self.n + 1):
            self.f[i][0] = self.lst[i - 1]
        for j in range(1, int(math.log2(self.n)) + 1):
            for i in range(1, self.n - (1 << j) + 2):
                a = self.f[i][j - 1]
                b = self.f[i + (1 << (j - 1))][j - 1]
                self.f[i][j] = a if a > b else b
        return

    def query(self, left, right):
        # 查询数组的索引left和right从1开始
        k = int(math.log2(right - left + 1))
        a = self.f[left][k]
        b = self.f[right - (1 << k) + 1][k]
        return a if a > b else b


class TestGeneral(unittest.TestCase):

    def test_skip_table(self):
        n = 8
        lst = [9, 3, 1, 7, 5, 6, 0, 8]
        st = SkipTable(n, lst)
        queries = [[1, 6], [1, 5], [2, 7], [2, 6], [1, 8], [4, 8], [3, 7], [1, 8]]
        assert [st.query(left, right) for left, right in queries] == [9, 9, 7, 7, 9, 8, 7, 9]
        return


if __name__ == '__main__':
    unittest.main()
