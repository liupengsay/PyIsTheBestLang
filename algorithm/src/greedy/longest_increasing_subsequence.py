"""

"""

"""
LIS：Longest Increasing Subsequence
问题1：最长单调递增子序列（严格上升）：<
问题2：最长单调不减子序列（不降）：<=
问题3：最长单调递减子序列（严格下降）：>
问题4：最长单调不增子序列（不升）：>=
对于数组来说，正数反可以将后两个问题3和4转换为前两个问题1和2进行解决，可以算全局的最长单调子序列，也可以计算前后缀的最长单调子序列

dilworth定理，不下降子序列最小个数等于最大上升子序列的长度，不上升子序列最小个数等于最大下降子序列的长度。
参考题目：
P1020 导弹拦截（https://www.luogu.com.cn/problem/P1020）使用贪心加二分计算最长单调不减和单调不增子序列的长度
P1439 最长公共子序列（https://www.luogu.com.cn/problem/P1439）使用贪心加二分计算最长单调递增子序列的长度
P1091 合唱队形（https://www.luogu.com.cn/problem/P1091）可以往前以及往后计算最长单调子序列
L2111 使数组 K 递增的最少操作次数（https://leetcode.cn/problems/minimum-operations-to-make-the-array-k-increasing/）分成 K 组计算每组的最长递增子序列
P1233 木棍加工（https://www.luogu.com.cn/problem/P1233）

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
import bisect


class LongestIncreasingSubsequence:
    def __init__(self):
        return

    @staticmethod
    def definitely_increase(nums):
        # 最长单调递增子序列（严格上升）
        dp = []
        for num in nums:
            i = bisect.bisect_left(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    @staticmethod
    def definitely_not_reduce(nums):
        # 最长单调不减子序列（不降）
        dp = []
        for num in nums:
            i = bisect.bisect_right(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    def definitely_reduce(self, nums):
        # 最长单调递减子序列（严格下降）
        nums = [-num for num in nums]
        return self.definitely_increase(nums)

    def definitely_not_increase(self, nums):
        # 最长单调不增子序列（不升）
        nums = [-num for num in nums]
        return self.definitely_not_reduce(nums)


class TestGeneral(unittest.TestCase):

    def test_longest_increasing_subsequence(self):
        lis = LongestIncreasingSubsequence()
        nums = [1, 2, 3, 3, 2, 2, 1]
        assert lis.definitely_increase(nums) == 3
        assert lis.definitely_not_reduce(nums) == 4
        assert lis.definitely_reduce(nums) == 3
        assert lis.definitely_not_increase(nums) == 5
        return


if __name__ == '__main__':
    unittest.main()
