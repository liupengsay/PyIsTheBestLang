"""

"""
"""
算法：深度优先搜索
功能：常与回溯枚举结合使用，比较经典的还有DFS序
题目：
P1120 小木棍（https://www.luogu.com.cn/problem/P1120）把数组分成和相等的子数组
P1692 部落卫队（https://www.luogu.com.cn/problem/P1692）暴力搜索枚举字典序最大可行的连通块

P1612 [yLOI2018] 树上的链（https://www.luogu.com.cn/problem/P1612）使用dfs记录路径的前缀和并使用二分确定最长链条
P1475 [USACO2.3]控制公司 Controlling Companies（https://www.luogu.com.cn/problem/P1475）深搜确定可以控制的公司对

P2080 增进感情（https://www.luogu.com.cn/problem/P2080）深搜回溯与剪枝
301. 删除无效的括号（https://leetcode.cn/problems/remove-invalid-parentheses/）深搜回溯与剪枝
P2090 数字对（https://www.luogu.com.cn/problem/P2090）深搜贪心回溯剪枝与辗转相减法

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


class DFS:
    def __init__(self):
        return

    @staticmethod
    def gen_node_order(dct):
        # 生成深搜序即 dfs 序以及对应子树编号区间
        def dfs(x):
            nonlocal order
            visit[x] = order
            order += 1
            for y in dct[x]:
                if not visit[y]:
                    dfs(y)
            interval[x] = [visit[x], order-1]
            return

        n = len(dct)
        order = 1
        visit = [0]*n
        interval = [[] for _ in range(n)]

        dfs(0)
        return visit, interval

    @staticmethod
    def add_to_n(n):

        # 计算将 [1, 1] 通过 [a, b] 到 [a, a+b] 或者 [a+b, a] 的方式最少次数变成 a == n or b == n
        if n == 1:
            return 0

        def gcd_minus(a, b, c):
            nonlocal ans
            if c >= ans or not b:
                return
            assert a >= b
            if b == 1:
                ans = ans if ans < c + a - 1 else c + a - 1
                return

            # 逆向思维计算保证使 b 减少到 a 以下
            gcd_minus(b, a % b, c + a // b)
            return

        ans = n - 1
        for i in range(1, n):
            gcd_minus(n, i, 0)
        return ans


class TestGeneral(unittest.TestCase):

    def test_dfs(self):
        dfs = DFS()
        dct = [[1, 2], [0, 3], [0, 4], [1], [2]]
        visit, interval = dfs.gen_node_order(dct)
        assert visit == [1, 2, 4, 3, 5]
        assert interval == [[1, 5], [2, 3], [4, 5], [3, 3], [5, 5]]
        return


if __name__ == '__main__':
    unittest.main()
