import math
import random
import unittest
from functools import reduce, lru_cache
from math import gcd
from operator import add
from itertools import accumulate
from typing import List
from operator import mul, add, xor, and_, or_
from algorithm.src.fast_io import FastIO

"""
算法：轮廓线DP
功能：
参考：
题目：

===================================力扣===================================
1659. 最大化网格幸福感（https://leetcode.cn/problems/maximize-grid-happiness/）轮廓线 DP 经典题目

===================================洛谷===================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

================================CodeForces================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1659_1(m: int, n: int, introverts: int, extroverts: int) -> int:
        # 模板：记忆化深搜进行轮廓线 DP
        def dfs(pos, state, intro, ext):
            # 当前网格位置，前 n 个格子压缩状态，剩余内向人数，剩余外向人数
            res = dp[pos][state][intro][ext]
            if res != -1:
                return res
            if not intro and not ext:
                dp[pos][state][intro][ext] = 0
                return 0
            if pos == m * n:
                dp[pos][state][intro][ext] = 0
                return 0
            res = dfs(pos + 1, 3 * (state % s), intro, ext)
            if intro:
                cur = dfs(pos + 1, 3 * (state % s) + 1, intro - 1, ext) + 120
                # 左边
                if pos % n:
                    cur += cross[state % 3][1]
                # 上边
                if pos // n:
                    cur += cross[state // s][1]
                res = res if res > cur else cur
            if ext:
                cur = dfs(pos + 1, 3 * (state % s) + 2, intro, ext - 1) + 40
                # 左边
                if pos % n:
                    cur += cross[state % 3][2]
                # 上边
                if pos // n:
                    cur += cross[state // s][2]
                res = res if res > cur else cur
            dp[pos][state][intro][ext] = res
            return res

        s = 3 ** (n - 1)
        cross = [[0, 0, 0], [0, -60, -10], [40, -10, 40]]
        # 手写记忆化进行内存优化
        dp = [[[[-1] * (extroverts + 1) for _ in range(introverts + 1)] for _ in range(s * 3)] for _ in range(m * n + 1)]
        return dfs(0, 0, introverts, extroverts)

    @staticmethod
    def lc_1659_2(m: int, n: int, introverts: int, extroverts: int) -> int:
        # 模板：迭代进行轮廓线 DP
        s = 3 ** (n - 1)
        cross = [[0, 0, 0], [0, -60, -10], [40, -10, 40]]
        dp = [[[[0] * (extroverts + 1) for _ in range(introverts + 1)] for _ in range(s * 3)] for _ in range(m * n + 1)]
        for pos in range(m * n - 1, -1, -1):
            # 还可以进行滚动数组优化
            for state in range(s * 3):
                for intro in range(introverts + 1):
                    for ext in range(extroverts + 1):
                        if intro == ext == 0:
                            continue
                        res = dp[pos + 1][3 * (state % s)][intro][ext]
                        if intro:
                            cur = dp[pos + 1][3 * (state % s) + 1][intro - 1][ext] + 120
                            # 左边
                            if pos % n:
                                cur += cross[state % 3][1]
                            # 上边
                            if pos // n:
                                cur += cross[state // s][1]
                            res = res if res > cur else cur
                        if ext:
                            cur = dp[pos + 1][3 * (state % s) + 2][intro][ext - 1] + 40
                            # 左边
                            if pos % n:
                                cur += cross[state % 3][2]
                            # 上边
                            if pos // n:
                                cur += cross[state // s][2]
                            res = res if res > cur else cur
                        dp[pos][state][intro][ext] = res
        return dp[0][0][introverts][extroverts]


class TestGeneral(unittest.TestCase):

    def test_xxxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
