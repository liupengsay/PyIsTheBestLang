"""

"""
"""
算法：背包DP、分组背包、一维（无限有限）背包、二位背包、多重背包、分组背包、限制背包
功能：一重背包DP，数量有限从后往前遍历，数量无限则从前往后遍历；多重背包DP，可使用二进制拆分进行优化。

题目：
L0214 最短回文串（https://leetcode.cn/problems/shortest-palindrome/）计算字符串前缀最长回文子串
L2218 从栈中取出 K 个硬币的最大面值和（https://leetcode.cn/problems/maximum-value-of-k-coins-from-piles/）背包DP
P1048 采药（https://www.luogu.com.cn/problem/P1048）一维背包DP，数量有限，从后往前遍历
P1049 [NOIP2001 普及组] 装箱问题（https://www.luogu.com.cn/problem/P1049）一维背包DP
P1776 宝物筛选（https://www.luogu.com.cn/problem/P1776）多重背包，使用二进制拆分进行优化
P1509 找啊找啊找GF（https://www.luogu.com.cn/problem/P1509）四重背包
P1060 [NOIP2006 普及组] 开心的金明（https://www.luogu.com.cn/problem/P1509）一维背包DP
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

    def main_p1776(self):

        import sys
        sys.setrecursionlimit(10000000)

        def read():
            return sys.stdin.readline()

        def ac(x):
            return sys.stdout.write(str(x) + '\n')

        # 二进制拆分
        def bin_split(num):
            res = []
            x = 1
            while num >= x:
                res.append(x)
                num -= x
                x *= 2
            if num:
                res.append(num)
            return res

        def main():
            n, w = map(int, read().split())
            pre = [0] * (w + 1)
            for _ in range(n):
                val, weight, amount = map(int, read().split())
                cur = pre[:]
                for x in bin_split(amount):
                    for i in range(w, x * weight - 1, -1):
                        c = cur[i - x * weight] + x * val
                        if cur[i] < c:
                            cur[i] = c
                pre = cur[:]
            ac(max(pre))
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
