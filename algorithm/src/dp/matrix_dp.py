"""

"""
"""
算法：矩阵DP
功能：在二维矩阵上进行转移的DP，经典的有矩阵前缀和，矩阵区间和，正方形最大边长或面积
题目：
L2435 矩阵中和能被 K 整除的路径（https://leetcode.cn/problems/paths-in-matrix-whose-sum-is-divisible-by-k/）利用模 K 的特点进行路径计算
L2088 统计农场中肥沃金字塔的数目（https://leetcode.cn/problems/count-fertile-pyramids-in-a-land/）类似求正方形的边长和面积进行矩阵DP
P1681 最大正方形II（https://www.luogu.com.cn/problem/P1681）求黑白格子相间的最大正方形面积
221. 最大正方形（https://leetcode.cn/problems/maximal-square/）求全为 1 的最大正方形面积
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


class MatrixDP:
    def __init__(self):
        return

    @staticmethod
    def maximal_square(matrix: List[List[str]]) -> int:

        # 求全为 1 的最大正方形面积
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        ans = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    # 转移公式
                    dp[i + 1][j + 1] = min(dp[i][j], dp[i + 1][j], dp[i][j + 1]) + 1
                    if dp[i + 1][j + 1] > ans:
                        ans = dp[i + 1][j + 1]
        return ans ** 2


class TestGeneral(unittest.TestCase):

    def test_matrix_dp(self):
        md = MatrixDP()
        matrix = [["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"], ["1", "0", "0", "1", "0"]]
        assert md.maximal_square(matrix) == 4
        return


if __name__ == '__main__':
    unittest.main()
