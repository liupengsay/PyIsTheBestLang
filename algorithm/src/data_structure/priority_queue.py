"""
算法：单调队列、双端队列
功能：维护单调性，计算滑动窗口最大值最小值
题目：
P2251 质量检测（https://www.luogu.com.cn/problem/P2251）滑动区间最小值
L0239 滑动窗口最大值（https://leetcode.cn/problems/sliding-window-maximum/）滑动区间最大值
参考：OI WiKi（xx）

P1750 出栈序列（https://www.luogu.com.cn/problem/P1750）经典题目，滑动指针窗口栈加队列
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


class PriorityQueue:
    def __init__(self):
        return

    @staticmethod
    def sliding_window(nums, k, method="max"):
        assert k >= 1
        # 计算滑动窗口最大值与最小值
        if method == "min":
            nums = [-num for num in nums]
        n = len(nums)
        stack = deque()
        ans = []
        for i in range(n):
            while stack and stack[0][1] <= i - k:
                stack.popleft()
            while stack and stack[-1][0] <= nums[i]:
                stack.pop()
            stack.append([nums[i], i])
            if i >= k - 1:
                ans.append(stack[0][0])
        if method == "min":
            ans = [-num for num in ans]
        return ans


class TestGeneral(unittest.TestCase):

    def test_priority_queue(self):
        pq = PriorityQueue()

        for _ in range(10):
            n = random.randint(100, 1000)
            nums = [random.randint(1, n) for _ in range(n)]
            k = random.randint(1, n)
            ans = pq.sliding_window(nums, k, "max")
            for i in range(n-k+1):
                assert ans[i] == max(nums[i:i+k])

            ans = pq.sliding_window(nums, k, "min")
            for i in range(n - k + 1):
                assert ans[i] == min(nums[i:i + k])
        return


if __name__ == '__main__':
    unittest.main()
