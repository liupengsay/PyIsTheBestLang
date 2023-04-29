
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


"""
算法：扩展欧几里得定理、extended_gcd、binary_gcd、二进制gcd、裴蜀定理
功能：用于求解单个同余方程
题目：

===================================力扣===================================
6301. 判断一个点是否可以到达（https://leetcode.cn/contest/biweekly-contest-96/problems/check-if-point-is-reachable/）binary_gcd快速求解，扩展欧几里得算法gcd为1判断可达性

===================================洛谷===================================
P1082 同余方程（https://www.luogu.com.cn/problem/P1082）转化为同余方程求解最小的正整数解
P5435 基于值域预处理的快速 GCD（https://www.luogu.com.cn/problem/P5435）binary_gcd快速求解
P5582 【SWTR-01】Escape（https://www.luogu.com.cn/problem/P5582）贪心加脑筋急转弯，使用扩展欧几里得算法gcd为1判断可达性
P1516 青蛙的约会（https://www.luogu.com.cn/problem/P1516）求解a*x+b*y=m的最小正整数解

参考：OI WiKi（xx）
"""


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

    def solve_equal(self, a, b, m=1):
        # 模板：扩展gcd求解ax+by=m方程组的所有解
        gcd, x0, y0 = self.extend_gcd(a, b)
        # 方程有解当且仅当c是gcd(a,b)的倍数
        assert a * x0 + b * y0 == 1

        # 方程组的解初始值则为
        x1 = x0 * (m // gcd)
        y1 = y0 * (m // gcd)

        # 所有解为下面这些，可进一步求解正数负数解
        # x = x1+b//gcd*t(t=0,1,2,3,...)
        # y = y1-a//gcd*t(t=0,1,2,3,...)
        return [gcd, x1, y1]

    @staticmethod
    def binary_gcd(a, b):
        # 模板：二进制gcd，使用二进制求两个正数的gcd
        assert a > 0 and b > 0
        c = 1
        while a - b:
            if a & 1:
                if b & 1:
                    if a > b:
                        a = (a - b) >> 1
                    else:
                        b = (b - a) >> 1
                else:
                    b = b >> 1
            else:
                if b & 1:
                    a = a >> 1
                else:
                    c = c << 1
                    b = b >> 1
                    a = a >> 1
        return c * a


class Solution:
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

        for _ in range(1000):
            a = random.randint(1, 10**9)
            b = random.randint(1, 10**9)
            assert ExtendGcd().binary_gcd(a, b) == math.gcd(a, b)
        return


if __name__ == '__main__':
    unittest.main()
