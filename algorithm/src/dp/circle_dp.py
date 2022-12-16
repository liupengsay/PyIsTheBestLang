"""

"""
"""
算法：环形线性或者区间DP
功能：计算环形数组上的操作，比较简单的方式是将数组复制成两遍进行区间或者线性DP

题目：
L1880 石子合并（https://www.luogu.com.cn/problem/P1880）将数组复制成两遍进行区间DP

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


class ClassName:
    def __init__(self):
        return

    def gen_result(self):
        return

    def main_p1880(self):
        import sys
        from functools import lru_cache
        sys.setrecursionlimit(10000000)

        def read():
            return sys.stdin.readline()

        def ac(x):
            return sys.stdout.write(str(x) + '\n')

        def main():
            n = int(read())
            nums = list(map(int, read().split()))
            nums = nums + nums
            n *= 2
            pre = [0] * (n + 1)
            for i in range(n):
                pre[i + 1] = pre[i] + nums[i]

            @lru_cache(None)
            def floor(x, y):
                if x >= y:
                    return 0
                if x == y - 1:
                    return nums[x] + nums[x + 1]
                res = float("inf")
                for k in range(x, y):
                    cur = floor(x, k) + floor(k + 1, y) + pre[y + 1] - pre[x]
                    res = res if res < cur else cur
                return res

            ac(min(floor(i, i + n // 2 - 1) for i in range(n // 2)))

            @lru_cache(None)
            def ceil(x, y):
                if x >= y:
                    return 0
                if x == y - 1:
                    return nums[x] + nums[x + 1]
                res = 0
                for k in range(x, y):
                    cur = ceil(x, k) + ceil(k + 1, y) + pre[y + 1] - pre[x]
                    res = res if res > cur else cur
                return res

            ac(max(ceil(i, i + n // 2 - 1) for i in range(n // 2)))
            return

        main()
        return


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        nt = ClassName()
        assert nt.gen_result(10 ** 11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
