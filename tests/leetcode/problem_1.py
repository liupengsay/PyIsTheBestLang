import bisect
import random
import re
import unittest

from typing import List, Callable
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal
from typing import List, Callable
import heapq
import copy
from sortedcontainers import SortedList


class SegmentTreeRangeAddMax:
    # 模板：线段树区间更新、持续增加最大值
    def __init__(self, n):
        self.floor = 0
        self.height = [self.floor]*(4*n)
        self.lazy = [self.floor]*(4*n)

    @staticmethod
    def max(a, b):
        return a if a > b else b

    def push_down(self, i):
        # 懒标记下放，注意取最大值
        if self.lazy[i]:
            self.height[2 * i] = self.max(self.height[2 * i], self.lazy[i])
            self.height[2 * i + 1] = self.max(self.height[2 * i + 1], self.lazy[i])
            self.lazy[2 * i] = self.max(self.lazy[2 * i], self.lazy[i])
            self.lazy[2 * i + 1] = self.max(self.lazy[2 * i + 1], self.lazy[i])
            self.lazy[i] = self.floor
        return

    def update(self, left, right, s, t, val, i):
        # 更新区间最大值
        stack = [[s, t, i]]
        while stack:
            a, b, i = stack.pop()
            if i >= 0:
                if left <= a and b <= right:
                    self.height[i] = self.max(self.height[i], val)
                    self.lazy[i] = self.max(self.lazy[i], val)
                    continue
                self.push_down(i)
                stack.append([a, b, ~i])
                m = a + (b - a) // 2
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([a, m, 2 * i])
                if right > m:
                    stack.append([m + 1, b, 2 * i + 1])
            else:
                i = ~i
                self.height[i] = self.max(self.height[2 * i], self.height[2 * i + 1])
        return

    def query(self, left, right, s, t, i):
        # 查询区间的最大值
        stack = [[s, t, i]]
        highest = self.floor
        while stack:
            a, b, i = stack.pop()
            if left <= a and b <= right:
                highest = self.max(highest, self.height[i])
                continue
            self.push_down(i)
            m = a + (b - a) // 2
            if left <= m:
                stack.append([a, m, 2*i])
            if right > m:
                stack.append([m+1, b, 2*i + 1])
        return highest


class Solution:
    def maxJumps(self, nums: List[int], d: int) -> int:

        # 经典单调栈灵活求解
        n = len(nums)
        post = [n-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                post[stack.pop()] = i - 1
            stack.append(i)

        pre = [0] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                pre[stack.pop()] = i + 1
            stack.append(i)

        dct = defaultdict(list)
        for i, num in enumerate(nums):
            dct[num].append(i)

        tree = SegmentTreeRangeAddMax(n)

        for num in sorted(dct):
            cur = []
            for i in dct[num]:
                left, right = pre[i], post[i]
                x = tree.query(left, right, 0, n-1, 1)
                cur.append([x+1, i])
            for x, i in cur:
                tree.update(i, i, 0, n-1, x, 1)
        return tree.query(0, n-1, 0, n-1, 1)

assert Solution().maxJumps(nums = [6,4,14,6,8,13,9,7,10,6,12], d = 2) == 4