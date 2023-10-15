import heapq
from bisect import insort_left, bisect_left
from collections import Counter, deque, defaultdict
from typing import List
from math import inf

from sortedcontainers import SortedList

from src.data_structure.sorted_list import LocalSortedList
from src.fast_io import FastIO
from src.mathmatics.number_theory import NumberTheory


import math
import unittest


class BrainStorming:
    def __init__(self):
        return

    @staticmethod
    def minimal_coin_need(n, m, nums):

        nums += [m + 1]
        nums.sort()
        # 有 n 个可选取且无限的硬币，为了形成 1-m 所有组合需要的最少硬币个数
        if nums[0] != 1:
            return -1
        ans = sum_ = 0
        for i in range(n):
            nex = nums[i + 1] - 1
            nex = nex if nex < m else m
            x = math.ceil((nex - sum_) / nums[i])
            x = x if x >= 0 else 0
            ans += x
            sum_ += x * nums[i]
            if sum_ >= m:
                break
        return ans
