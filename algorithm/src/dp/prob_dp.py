
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

from algorithm.src.fast_io import FastIO

"""
算法：概率DP
功能：根据组合数与转移方案求解概率或者期望
题目：

===================================洛谷===================================
P2719 搞笑世界杯（https://www.luogu.com.cn/record/list?user=739032&status=12&page=1）二维DP求概率
P1291 [SHOI2002] 百事世界杯之旅（https://www.luogu.com.cn/problem/P1291）线性DP求期望

参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):

        # 模板：经典记忆化二维 DP 模拟搜索转移计算概率

        @lru_cache(None)
        def dfs(a, b):
            if a + b == 2:
                if a == 0 or b == 0:
                    return 1
                return 0
            if a == 0 or b == 0:
                return 1
            res = (dfs(a - 1, b) + dfs(a, b - 1)) / 2
            return res

        n = ac.read_int() // 2
        if n == 0:
            ans = 0
        else:
            ans = dfs(n, n)
        ac.st("%.4f" % ans)
        return

    @staticmethod
    def lg_p1291(ac=FastIO()):

        # 模板：线性DP求期望，使用分数加减运算
        n = ac.read_int()
        ans = [1, 1]
        for x in range(2, n+1):
            a, b = ans
            c, d = 1, x
            g = math.gcd(b, d)
            lcm = b*d//g
            a, b = a*lcm//b+c*lcm//d, lcm
            g = math.gcd(a, b)
            ans = [a//g, b//g]

        a, b = ans
        a *= n
        x = a//b
        a %= b
        if a == 0:
            ac.st(x)
            return
        g = math.gcd(a, b)
        ans = [a//g, b//g]
        a, b = ans
        ac.st(len(str(x))*" " + str(a))
        ac.st(str(x)+"-"*len(str(b)))
        ac.st(len(str(x))*" " + str(b))
        return


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
