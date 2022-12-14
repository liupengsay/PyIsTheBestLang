"""

"""
"""
算法：线段树
功能：用以修改和查询区间的值信息，支持增减、修改，区间和、区间最大值、区间最小值
题目：xx（xx）

P3372 线段树（https://www.luogu.com.cn/problem/P3372）区间值增减与计算区间和
L0218 天际线问题（https://leetcode.cn/problems/the-skyline-problem/solution/by-liupengsay-isfo/）区间值修改与计算最大值
L2286 以组为单位订音乐会的门票（https://leetcode.cn/problems/booking-concert-tickets-in-groups/）区间值增减与计算区间和、区间最大值、区间最小值

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


class SegmentTreeRangeAddSum:
    def __init__(self):
        # 区间值增减与区间和查询
        self.cover = defaultdict(int)
        self.lazy = defaultdict(int)

    def push_down(self, i, s, m, t):
        if self.lazy[i]:
            self.cover[2 * i] += self.lazy[i]*(m-s+1)
            self.cover[2 * i + 1] += self.lazy[i]*(t-m)

            self.lazy[2 * i] += self.lazy[i]
            self.lazy[2 * i + 1] += self.lazy[i]

            self.lazy[i] = 0

    def update(self, left, r, s, t, val, i):
        if left <= s and t <= r:
            self.cover[i] += val*(t-s+1)
            self.lazy[i] += val
            return
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        if left <= m:
            self.update(left, r, s, m, val, 2 * i)
        if r > m:
            self.update(left, r, m + 1, t, val, 2 * i + 1)
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query(self, left, r, s, t, i):
        if left <= s and t <= r:
            return self.cover[i]
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        ans = 0
        if left <= m:
            ans += self.query(left, r, s, m, 2 * i)
        if r > m:
            ans += self.query(left, r, m + 1, t, 2 * i + 1)
        return ans


class SegmentTreeRangeUpdateSum:
    def __init__(self):
        # 区间值修改与区间和查询
        self.cover = defaultdict(int)
        self.lazy = defaultdict(int)

    def push_down(self, i, s, m, t):
        if self.lazy[i]:
            self.cover[2 * i] = self.lazy[i]*(m-s+1)
            self.cover[2 * i + 1] = self.lazy[i]*(t-m)

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = 0

    def update(self, left, r, s, t, val, i):
        if left <= s and t <= r:
            self.cover[i] = val*(t-s+1)
            self.lazy[i] = val
            return
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        if left <= m:
            self.update(left, r, s, m, val, 2 * i)
        if r > m:
            self.update(left, r, m + 1, t, val, 2 * i + 1)
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query(self, left, r, s, t, i):
        if left <= s and t <= r:
            return self.cover[i]
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        ans = 0
        if left <= m:
            ans += self.query(left, r, s, m, 2 * i)
        if r > m:
            ans += self.query(left, r, m + 1, t, 2 * i + 1)
        return ans


class SegmentTreePointAddSumMaxMin:
    def __init__(self, n: int):
        # 索引从 1 开始
        self.n = n
        self.min = [0] * (n * 4)
        self.sum = [0] * (n * 4)
        self.max = [0] * (n * 4)

    # 将 idx 上的元素值增加 val
    def add(self, o: int, l: int, r: int, idx: int, val: int) -> None:
        # 索引从 1 开始
        if l == r:
            self.min[o] += val
            self.sum[o] += val
            self.max[o] += val
            return
        m = (l + r) // 2
        if idx <= m:
            self.add(o * 2, l, m, idx, val)
        else:
            self.add(o * 2 + 1, m + 1, r, idx, val)
        self.min[o] = min(self.min[o * 2], self.min[o * 2 + 1])
        self.max[o] = max(self.max[o * 2], self.max[o * 2 + 1])
        self.sum[o] = self.sum[o * 2] + self.sum[o * 2 + 1]

    # 返回区间 [L,R] 内的元素和
    def query_sum(self, o: int, l: int, r: int, L: int, R: int) -> int:
        # 索引从 1 开始
        if L <= l and r <= R:
            return self.sum[o]
        sum_ = 0
        m = (l + r) // 2
        if L <= m:
            sum_ += self.query_sum(o * 2, l, m, L, R)
        if R > m:
            sum_ += self.query_sum(o * 2 + 1, m + 1, r, L, R)
        return sum_

    # 返回区间 [L,R] 内的最小值
    def query_min(self, o: int, l: int, r: int, L: int, R: int) -> int:
        # 索引从 1 开始
        if L <= l and r <= R:
            return self.min[o]
        res = 10 ** 9 + 7
        m = (l + r) // 2
        if L <= m:
            res = min(res, self.query_min(o * 2, l, m, L, R))
        if R > m:
            res = min(res, self.query_min(o * 2 + 1, m + 1, r, L, R))
        return res

    # 返回区间 [L,R] 内的最大值
    def query_max(self, o: int, l: int, r: int, L: int, R: int) -> int:
        # 索引从 1 开始
        if L <= l and r <= R:
            return self.max[o]
        res = 0
        m = (l + r) // 2
        if L <= m:
            res = max(res, self.query_max(o * 2, l, m, L, R))
        if R > m:
            res = max(res, self.query_max(o * 2 + 1, m + 1, r, L, R))
        return res


class SegmentTreeRangeAddMax:
    # 持续增加最大值
    def __init__(self):
        self.height = defaultdict(int)
        self.lazy = defaultdict(int)

    def push_down(self, i):
        # 懒标记下放，注意取最大值
        if self.lazy[i]:
            self.height[2 * i] = self.height[2 * i] if self.height[2 * i] > self.lazy[i] else self.lazy[i]
            self.height[2 * i + 1] = self.height[2 * i + 1] if self.height[2 * i + 1] > self.lazy[i] else self.lazy[i]

            self.lazy[2 * i] = self.lazy[2 * i] if self.lazy[2 * i] > self.lazy[i] else self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[2 * i + 1] if self.lazy[2 * i + 1] > self.lazy[i] else self.lazy[i]

            self.lazy[i] = 0
        return

    def update(self, l, r, s, t, val, i):
        # 更新区间最大值
        if l <= s and t <= r:
            self.height[i] = self.height[i] if self.height[i] > val else val
            self.lazy[i] = self.lazy[i] if self.lazy[i] > val else val
            return
        self.push_down(i)
        m = s + (t - s) // 2
        if l <= m:  # 注意左右子树的边界与范围
            self.update(l, r, s, m, val, 2 * i)
        if r > m:
            self.update(l, r, m + 1, t, val, 2 * i + 1)
        self.height[i] = self.height[2 * i] if self.height[2 * i] > self.height[2 * i + 1] else self.height[2 * i + 1]
        return

    def query(self, l, r, s, t, i):
        # 查询区间的最大值
        if l <= s and t <= r:
            return self.height[i]
        self.push_down(i)
        m = s + (t - s) // 2
        highest = float("-inf")
        if l <= m:
            cur = self.query(l, r, s, m, 2 * i)
            if cur > highest:
                highest = cur
        if r > m:
            cur = self.query(l, r, m + 1, t, 2 * i + 1)
            if cur > highest:
                highest = cur
        return highest


class SegmentTreeRangeUpdateMax:
    # 持续修改区间值
    def __init__(self):
        self.height = defaultdict(lambda: float("-inf"))
        self.lazy = defaultdict(int)

    def push_down(self, i):
        # 懒标记下放，注意取最大值
        if self.lazy[i]:
            self.height[2 * i] = self.lazy[i]
            self.height[2 * i + 1] = self.lazy[i]

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = 0
        return

    def update(self, l, r, s, t, val, i):
        # 更新区间最大值
        if l <= s and t <= r:
            self.height[i] = val
            self.lazy[i] = val
            return
        self.push_down(i)
        m = s + (t - s) // 2
        if l <= m:  # 注意左右子树的边界与范围
            self.update(l, r, s, m, val, 2 * i)
        if r > m:
            self.update(l, r, m + 1, t, val, 2 * i + 1)
        self.height[i] = self.height[2 * i] if self.height[2 * i] > self.height[2 * i + 1] else self.height[2 * i + 1]
        return

    def query(self, l, r, s, t, i):
        # 查询区间的最大值
        if l <= s and t <= r:
            return self.height[i]
        self.push_down(i)
        m = s + (t - s) // 2
        highest = float("-inf")
        if l <= m:
            cur = self.query(l, r, s, m, 2 * i)
            if cur > highest:
                highest = cur
        if r > m:
            cur = self.query(l, r, m + 1, t, 2 * i + 1)
            if cur > highest:
                highest = cur
        return highest


class SegmentTreeRangeUpdateMin:
    # 持续修改区间值
    def __init__(self):
        self.height = defaultdict(lambda: float("inf"))
        self.lazy = defaultdict(int)

    def push_down(self, i):
        # 懒标记下放，注意取最大值
        if self.lazy[i]:
            self.height[2 * i] = self.lazy[i]
            self.height[2 * i + 1] = self.lazy[i]

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = 0
        return

    def update(self, l, r, s, t, val, i):
        # 更新区间最大值
        if l <= s and t <= r:
            self.height[i] = val
            self.lazy[i] = val
            return
        self.push_down(i)
        m = s + (t - s) // 2
        if l <= m:  # 注意左右子树的边界与范围
            self.update(l, r, s, m, val, 2 * i)
        if r > m:
            self.update(l, r, m + 1, t, val, 2 * i + 1)
        self.height[i] = self.height[2 * i] if self.height[2 * i] < self.height[2 * i + 1] else self.height[2 * i + 1]
        return

    def query(self, l, r, s, t, i):
        # 查询区间的最大值
        if l <= s and t <= r:
            return self.height[i]
        self.push_down(i)
        m = s + (t - s) // 2
        highest = float("inf")
        if l <= m:
            cur = self.query(l, r, s, m, 2 * i)
            if cur < highest:
                highest = cur
        if r > m:
            cur = self.query(l, r, m + 1, t, 2 * i + 1)
            if cur < highest:
                highest = cur
        return highest


class SegmentTreeRangeAddMin:
    def __init__(self):
        # 持续减小最小值
        self.height = defaultdict(lambda: float("inf"))
        self.lazy = defaultdict(lambda: float("inf"))

    def push_down(self, i):
        # 懒标记下放，注意取最小值
        if self.lazy[i] < float("inf"):
            self.height[2*i] = self.height[2*i] if self.height[2*i] < self.lazy[i] else self.lazy[i]
            self.height[2*i+1] = self.height[2*i+1] if self.height[2*i+1] < self.lazy[i] else self.lazy[i]

            self.lazy[2*i] = self.lazy[2*i] if self.lazy[2*i] < self.lazy[i] else self.lazy[i]
            self.lazy[2*i+1] = self.lazy[2*i+1] if self.lazy[2*i+1] < self.lazy[i] else self.lazy[i]

            self.lazy[i] = float("inf")
        return

    def update(self, l, r, s, t, val, i):
        # 更新区间最小值
        if l<=s and t<=r:
            self.height[i] = self.height[i] if self.height[i] < val else val
            self.lazy[i] = self.lazy[i] if self.lazy[i] < val else val
            return
        self.push_down(i)
        m = s+(t-s)//2
        if l<=m: # 注意左右子树的边界与范围
            self.update(l, r, s, m, val, 2*i)
        if r>m:
            self.update(l, r, m+1, t, val, 2*i+1)
        self.height[i] = self.height[2*i] if self.height[2*i] < self.height[2*i+1] else self.height[2*i+1]
        return

    def query(self, l, r, s, t, i):
        # 查询区间的最小值
        if l<=s and t<=r:
            return self.height[i]
        self.push_down(i)
        m = s+(t-s)//2
        highest = float("inf")
        if l<=m:
            cur = self.query(l, r, s, m, 2*i)
            if cur < highest:
                highest = cur
        if r>m:
            cur = self.query(l, r, m+1, t, 2*i+1)
            if cur < highest:
                highest = cur
        return highest if highest < float("inf") else -1


class TestGeneral(unittest.TestCase):

    def test_segment_tree_range_add_sum(self):
        low = 0
        high = 10**9 + 7
        n = 1000
        nums = [random.randint(low, high) for _ in range(n)]
        stra = SegmentTreeRangeAddSum()
        for i in range(n):
            stra.update(i, i, low, high, nums[i], 1)

        for _ in range(n):
            # 区间增加值
            left = random.randint(0, n-1)
            right = random.randint(left, n-1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right+1):
                nums[i] += num
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            assert stra.query(left, right, low, high, 1) == sum(nums[left:right+1])

            # 单点增加值
            left = random.randint(0, n - 1)
            right =left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert stra.query(left, right, low, high, 1) == sum(nums[left:right + 1])
        return

    def test_segment_tree_range_update_sum(self):
        low = 0
        high = 10**9 + 7
        n = 1000
        nums = [random.randint(low, high) for _ in range(n)]
        stra = SegmentTreeRangeUpdateSum()
        for i in range(n):
            stra.update(i, i, low, high, nums[i], 1)

        for _ in range(n):
            # 区间增加值
            left = random.randint(0, n-1)
            right = random.randint(left, n-1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right+1):
                nums[i] = num
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            assert stra.query(left, right, low, high, 1) == sum(nums[left:right+1])

            # 单点增加值
            left = random.randint(0, n - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query(left, right, low, high, 1) == sum(nums[left:right + 1])
        return

    def test_segment_tree_point_add_sum_max_min(self):
        low = 0
        high = 1000

        nums = [random.randint(low, high) for _ in range(high)]
        staasmm = SegmentTreePointAddSumMaxMin(high)
        for i in range(high):
            staasmm.add(1, low, high, i+1, nums[i])

        for _ in range(high):
            # 单点进行增减值
            i = random.randint(0, high - 1)
            num = random.randint(low, high)
            nums[i] += num
            staasmm.add(1, low, high, i+1, num)

            # 查询区间和、最大值、最小值
            left = random.randint(0, high-1)
            right = random.randint(left, high-1)
            assert staasmm.query_sum(1, low, high, left+1, right+1) == sum(nums[left:right + 1])
            assert staasmm.query_max(1, low, high, left+1, right+1) == max(nums[left:right + 1])
            assert staasmm.query_min(1, low, high, left+1, right+1) == min(nums[left:right + 1])

            # 查询单点和、最大值、最小值
            left = random.randint(0, high - 1)
            right = left
            assert staasmm.query_sum(1, low, high, left+1, right+1) == sum(nums[left:right + 1])
            assert staasmm.query_max(1, low, high, left+1, right+1) == max(nums[left:right + 1])
            assert staasmm.query_min(1, low, high, left+1, right+1) == min(nums[left:right + 1])

        return

    def test_segment_tree_range_add_max(self):
        low = 0
        high = 1000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeAddMax()
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high-1)
            right = random.randint(left, high-1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right+1):
                nums[i] = nums[i] if nums[i] > num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(nums[left:right+1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num
            assert stra.query(left, right, low, high, 1) == max(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(nums[left:right + 1])
        return

    def test_segment_tree_range_update_max(self):
        low = 0
        high = 1000
        nums = [random.randint(low+1, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateMax()
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high-1)
            right = random.randint(left, high-1)
            num = random.randint(low+1, high)
            for i in range(left, right+1):
                nums[i] = num
            stra.update(left, right, low, high, num, 1)
            assert stra.query(left, right, low, high, 1) == max(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(nums[left:right+1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low+1, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query(left, right, low, high, 1) == max(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(nums[left:right + 1])
        return

    def test_segment_tree_range_update_min(self):
        low = 0
        high = 1000
        nums = [random.randint(low+1, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateMin()
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == min(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high-1)
            right = random.randint(left, high-1)
            num = random.randint(low+1, high)
            for i in range(left, right+1):
                nums[i] = num
            stra.update(left, right, low, high, num, 1)
            assert stra.query(left, right, low, high, 1) == min(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == min(nums[left:right+1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low+1, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query(left, right, low, high, 1) == min(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == min(nums[left:right + 1])
        return

    def test_segment_tree_range_add_min(self):
        low = 0
        high = 1000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeAddMin()
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == min(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high-1)
            right = random.randint(left, high-1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right+1):
                nums[i] = nums[i] if nums[i] < num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == min(nums[left:right+1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            assert stra.query(left, right, low, high, 1) == min(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == min(nums[left:right + 1])
        return


if __name__ == '__main__':
    unittest.main()
