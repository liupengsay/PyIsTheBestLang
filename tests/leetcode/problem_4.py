import bisect
import random
import re
import unittest

from typing import List, Callable, Dict
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
from typing import List, Callable
from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList





class SegmentTreeRangeUpdateQuerySumMinMax:
    def __init__(self, n) -> None:
        # 模板：区间值增减、区间和查询、区间最小值查询、区间最大值查询
        self.inf = inf
        self.n = n
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [0] * (4 * self.n)  # 懒标记
        self.floor = [0] * (4 * self.n)  # 最小值
        return

    @staticmethod
    def max(a: int, b: int) -> int:
        return a if a > b else b

    @staticmethod
    def min(a: int, b: int) -> int:
        return a if a < b else b

    def push_down(self, i: int, s: int, m: int, t: int) -> None:
        # 下放懒标记
        if self.lazy[i]:
            self.cover[2 * i] += self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] += self.lazy[i] * (t - m)

            self.floor[2 * i] += self.lazy[i]
            self.floor[2 * i + 1] += self.lazy[i]

            self.lazy[2 * i] += self.lazy[i]
            self.lazy[2 * i + 1] += self.lazy[i]

            self.lazy[i] = 0

    def push_up(self, i) -> None:
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.floor[i] = self.min(self.floor[2 * i], self.floor[2 * i + 1])
        return

    def make_tag(self, i, s, t, val) -> None:
        self.cover[i] += val * (t - s + 1)
        self.floor[i] += val
        self.lazy[i] += val
        return

    def update_range(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.make_tag(i, s, t, val)
                    continue

                m = s + (t - s) // 2
                self.push_down(i, s, m, t)
                stack.append([s, t, ~i])

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append([s, m, 2 * i])
                if right > m:
                    stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self.push_up(i)
        return

    def update_point(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减单点值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始

        while True:
            if left <= s and t <= right:
                self.make_tag(i, s, t, val)
                break
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        while i > 1:
            i //= 2
            self.push_up(i)
        return

    def query_sum(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return ans

    def query_min(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的最小值
        stack = [[s, t, i]]
        highest = self.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self.min(highest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self.push_down(i, s, m, t)
            if left <= m:
                stack.append([s, m, 2 * i])
            if right > m:
                stack.append([m + 1, t, 2 * i + 1])
        return highest


class BinarySearch:
    def __init__(self):
        return

    @staticmethod
    def find_int_left(low: int, high: int, check: Callable) -> int:
        # 模板: 整数范围内二分查找，选择最靠左满足check
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_int_right(low: int, high: int, check: Callable) -> int:
        # 模板: 整数范围内二分查找，选择最靠右满足check
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        return high if check(high) else low

    @staticmethod
    def find_float_left(low: float, high: float, check: Callable, error=1e-6) -> float:
        # 模板: 浮点数范围内二分查找, 选择最靠左满足check
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_float_right(low: float, high: float, check: Callable, error=1e-6) -> float:
        # 模板: 浮点数范围内二分查找, 选择最靠右满足check
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                low = mid
            else:
                high = mid
        return high if check(high) else low


class BookMyShow:

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.tree = SegmentTreeRangeUpdateQuerySumMinMax(n)
        self.cnt = [0]*n
        self.null = SortedList(list(range(n)))


    def gather(self, k: int, maxRow: int) -> List[int]:
        low = self.tree.query_min(0, maxRow-1, 0, self.n-1, 1)
        if self.m - low < k:
            return []

        def check(x):
            return self.m - self.tree.query_min(0, x, 0, self.n - 1, 1) >= k

        y = BinarySearch().find_int_left(0, maxRow-1, check)
        self.cnt[y] += k
        self.tree.update_point(y, y, 0, self.n-1, k, 1)
        if self.cnt[y] == self.m:
            self.null.discard(y)
        return [y, self.cnt[y]-k+1]

    def scatter(self, k: int, maxRow: int) -> bool:
        s = self.tree.query_sum(0, maxRow - 1, 0, self.n - 1, 1)
        if self.m*maxRow - s < k:
            return False
        while k:
            x = self.null[0]
            rest = k if k < self. m - self.cnt[x] else self. m - self.cnt[x]
            k -= rest
            self.cnt[x] += rest
            self.tree.update_point(x, x, 0, self.n-1, rest, 1)
            if self.cnt[x] == self.m:
                self.null.pop(0)
        return True





assert Solution().deleteString(st) == 4000
