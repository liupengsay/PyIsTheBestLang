import bisect
import unittest
from functools import lru_cache
from typing import List

from utils.fast_io import FastIO, inf
from collections import Counter, defaultdict, deque

from src.mathmatics.number_theory import NumberTheory


class LinearDP:
    def __init__(self):
        return

    @staticmethod
    def liner_dp_template(nums):
        # 线性 DP 递推模板（以最长上升子序列长度为例）
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = 1
            for j in range(i):
                if nums[i] > nums[j] and dp[j] + 1 > dp[i + 1]:
                    dp[i + 1] = dp[j] + 1
        return max(dp)


