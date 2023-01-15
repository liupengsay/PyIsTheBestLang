
import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from operator import xor, mul, add
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy


class Solution:
    def maxOutput(self,
                  n: int,
                  edges: List[List[int]],
                  price: List[int]) -> int:

        def mmax(a, b):
            return a if a > b else b

        def dfs(i, fa):
            nonlocal ans
            cur = []
            for j in dct[i]:
                if j != fa:
                    dfs(j, i)
                    cur.append(lst[j])

            # 计算所有子树的前缀后缀两项最大值
            m = len(cur)
            pre_0 = [0] * (m + 1)
            for x in range(m):
                pre_0[x + 1] = mmax(pre_0[x], cur[x][0])

            post_0 = [0] * (m + 1)
            for x in range(m - 1, -1, -1):
                post_0[x] = mmax(post_0[x + 1], cur[x][0])

            m = len(cur)
            pre_1 = [0] * (m + 1)
            for x in range(m):
                pre_1[x + 1] = mmax(pre_1[x], cur[x][1])

            post_1 = [0] * (m + 1)
            for x in range(m - 1, -1, -1):
                post_1[x] = mmax(post_1[x + 1], cur[x][1])

            # 超过两个子节点，选取不同最长路作为端点
            if m >= 2:
                for x in range(m):
                    y = mmax(pre_0[x], post_0[x + 1]) + cur[x][1] + price[i]
                    ans = ans if ans > y else y
                    y = mmax(pre_1[x], post_1[x + 1]) + cur[x][0] + price[i]
                    ans = ans if ans > y else y
            elif m == 1:  # 只有一个的时候 i 作为其中一个端点
                y = mmax(cur[0][1], cur[0][0] + price[i])
                ans = ans if ans > y else y
            if not cur:  # 没有子树
                lst[i] = [0, price[i]]
            else:  # 有子树，两项都加 price[i]
                lst[i] = [pre_0[-1] + price[i], pre_1[-1] + price[i]]
            return

        dct = [[] for _ in range(n)]
        for u, v in edges:
            dct[u].append(v)
            dct[v].append(u)
        # 树形 DP 记录 【子树去掉叶子节点的最长路，子树的最长路】
        lst = [[0, 0] for _ in range(n)]
        ans = 0
        dfs(0, -1)
        return ans


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().maxOutput(
            n=4, edges=[[2, 0], [0, 1], [1, 3]], price=[2, 3, 1, 1]) == 6
        return


if __name__ == '__main__':
    unittest.main()
