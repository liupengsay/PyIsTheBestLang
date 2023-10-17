import math
import random
import unittest
from collections import defaultdict, deque, Counter
from functools import lru_cache
from itertools import combinations
from typing import List
from math import inf
from src.graph.union_find import UnionFind
from src.mathmatics.number_theory import NumberTheory
from utils.fast_io import FastIO




class BagDP:
    def __init__(self):
        return

    @staticmethod
    def bin_split_1(num):
        # 二进制优化是指 1.2.4.x这样连续的而不是二进制10101对应的1
        if not num:
            return []
        lst = []
        x = 1
        while x <= num:
            lst.append(x)
            num -= x
            x *= 2
        if num:
            lst.append(num)
        return lst

    @staticmethod
    def bin_split_2(num):
        # 从大到小拆分保证没有 1 以外的相同正数
        if not num:
            return []
        lst = []
        while num:
            lst.append((num + 1) // 2)
            num //= 2
        lst.reverse()
        return lst

    @staticmethod
    def one_dimension_limited(n, nums):
        # 一维有限背包
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for i in range(n, num - 1, -1):
                dp[i] += dp[i - num]
        return dp[n]

    @staticmethod
    def one_dimension_unlimited(n, nums):
        # 一维无限背包
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for i in range(num, n + 1):
                dp[i] += dp[i - num]
        return dp[n]

    @staticmethod
    def two_dimension_limited(m, n, nums):
        # 二维有限背包（多维背包类似）
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        for a, b in nums:
            for i in range(m, a - 1, -1):
                for j in range(n, b - 1, -1):
                    dp[i][j] += dp[i - a][j - b]
        return dp[m][n]

    @staticmethod
    def two_dimension_unlimited(m, n, nums):
        # 二维无限背包（多维背包类似）
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        for a, b in nums:
            for i in range(a, m + 1):
                for j in range(b, n + 1):
                    dp[i][j] += dp[i - a][j - b]
        return dp[m][n]

    def continuous_bag_with_bin_split(self, n, nums):
        # 使用二进制优化的连续背包（以一维有限背包为例）
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for x in self.bin_split_1(num):
                for i in range(n, x - 1, -1):
                    dp[i] += dp[i - x]
        return dp[n]

    @staticmethod
    def group_bag_limited(n, d, nums):
        # 分组背包（以一维有限背包为例）计算出租车的最小花费
        pre = [inf] * (n + 1)
        pre[0] = 0
        for r, z in nums:
            cur = pre[:]  # 关键在于这里需要分组背包
            for x in range(1, z + 1):
                cost = d + x * r
                for i in range(n, x - 1, -1):
                    if pre[i - x] + cost < cur[i]:
                        cur[i] = pre[i - x] + cost
            pre = cur[:]
        if pre[n] < inf:
            return pre[n]
        return -1

    @staticmethod
    def group_bag_unlimited(nums):
        # 分组背包（以一维无限背包为例）计算 n 分解成四个数的平方和的方案数
        n = max(nums)
        dp = [[0] * 5 for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, int(math.sqrt(n)) + 1):
            x = i * i
            for j in range(x, n + 1):
                for k in range(1, 5):
                    if dp[j - x][k - 1]:
                        dp[j][k] += dp[j - x][k - 1]
        return [sum(dp[num]) for num in nums]

    @staticmethod
    def one_dimension_limited_use_dct(nums):
        # 一维有限背包（带负数的情况下使用字典做转移记录）
        pre = defaultdict(lambda: -inf)
        pre[0] = 0
        for s, f in nums:
            cur = pre.copy()
            for p in pre:
                cur[p + s] = max(cur[p + s], pre[p] + f)
            pre = cur
        ans = 0
        for p in pre:
            if p >= 0 and pre[p] >= 0:
                ans = ans if ans > p + pre[p] else p + pre[p]
        return ans




