"""

"""
"""
算法：扩展欧几里得定理
功能：用于求解单个同余方程
题目：
LP1082 同余方程（https://www.luogu.com.cn/problem/P1082）转化为同余方程求解最小的正整数解

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


class ExtendGcd:
    def __init__(self):
        return

    def extend_gcd(self, a, b):
        # 扩展欧几里得算法，求解ax+by=1，返回gcd(a,b)与x与y
        if a == 0:
            return b, 0, 1
        else:
            gcd, x, y = self.extend_gcd(b % a, a)
            return gcd, y - (b // a) * x, x

    def solve_equal(self, a, b, m = 1):
        # 求解ax+by=m方程组的所有解
        gcd, x0, y0 = self.extend_gcd(a, b)
        assert a * x0 + b * y0 == 1

        # 方程组的解初始值则为
        x1 = x0 * (m // gcd)
        y1 = y0 * (m // gcd)

        # 所有解为下面这些，可进一步求解正数负数解
        # x = x1+b//gcd*t(t=0,1,2,3,...)
        # y = y1-a//gcd*t(t=0,1,2,3,...)
        return [gcd, x1, y1]


class Luogu:
    def __init__(self):
        return

    @staticmethod
    def main_1082(a, b):
        gcd, x1, y1 = ExtendGcd().extend_gcd(a, b)
        # 求解最小的正整数解
        t = math.ceil(-x1 / (b / gcd))
        x = x1 + b // gcd * t
        if x <= 0:
            x = x1 + b // gcd * (t + 1)
        return x


class TestGeneral(unittest.TestCase):

    def test_extend_gcd(self):
        luogu = Luogu()
        assert luogu.main_1082(3, 10) == 7
        return


if __name__ == '__main__':
    unittest.main()
