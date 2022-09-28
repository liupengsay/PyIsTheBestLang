

import bisect
import random

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict

from functools import lru_cache

import random
from itertools import permutations
import numpy as np

from decimal import Decimal

import heapq
import copy



# 单点更新与前缀最小值
class TreeArray:
    # 数组索引从1开始
    def __init__(self, n):
        self.n = n
        self.c = [float("inf")]*(n+1)

    # 求x的二进制表示中，最低位的1的位置对应的数，向右相加更新管辖值，向左相减获得前缀和
    @staticmethod
    def lowest_bit(x):
        return x & -x

    # 给nums索引x增加k，同时维护对应受到影响的区间和c数组
    def add(self, x, k):
        while x <= self.n:  # 不能越界
            self.c[x] = min(self.c[x], k)
            x = x + self.lowest_bit(x)
        return

    # 前缀求和
    def get_sum(self, x):  # a[1]..a[x]的和
        ans = float("inf")
        while x >= 1:
            ans = min(ans, self.c[x])
            x -= self.lowest_bit(x)
        return ans


class Solution:
    def minJump(self, jump: List[int]) -> int:
        n = len(jump)
        segment_tree = TreeArray(n+1)
        segment_tree.add(1,1)
        for i in range(n):
            fast = min(i+jump[i], n)
            cur = segment_tree.c[i+1]
            segment_tree.add(fast+1, cur+1)
            segment_tree.add(0, fast, 0, n, cur + 2, 1)
        return segment_tree.c[n]

def test_solution():
    assert Solution().maximumANDSum([1, 2, 3, 4, 5, 6], 3) == 9
    return


if __name__ == '__main__':
    test_solution()
