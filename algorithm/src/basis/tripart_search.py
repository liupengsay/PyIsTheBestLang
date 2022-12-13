"""
算法：三分查找
功能：用来寻找区间至多具有一个峰顶点或者一个谷底点的函数极值解
题目：
P3382 三分法（https://www.luogu.com.cn/problem/P3382）利用三分求区间函数极值点
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


class TriPartSearch:
    def __init__(self):
        return

    @staticmethod
    def find_ceil(fun, left, right):
        error = 1e-6
        while left < right - error:
            diff = (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 > dist2:
                right = mid2
            elif dist1 < dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_floor(fun, left, right):
        error = 1e-6
        while left < right - error:
            diff = (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 < dist2:
                right = mid2
            elif dist1 > dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left


class TriPartPackTriPart:
    def __init__(self):
        return

    @staticmethod
    def find_floor(target, left_x, right_x, low_y, high_y):
        # 求最小的坐标[x,y]使得目标函数target最小
        error = 5e-8
        
        def optimize(y):
            # 套三分里面的损失函数
            low_ = left_x
            high_ = right_x
            while low_ < high_ - error:
                diff_ = (high_ - low_) / 3
                mid1_ = low_ + diff_
                mid2_ = low_ + 2 * diff_
                dist1_ = target(mid1_, y)
                dist2_ = target(mid2_, y)
                if dist1_ < dist2_:
                    high_ = mid2_
                elif dist1_ > dist2_:
                    low_ = mid1_
                else:
                    low_ = mid1_
                    high_ = mid2_
            return low_, target(low_, y)

        low = low_y
        high = high_y
        while low < high - error:
            diff = (high - low) / 3
            mid1 = low + diff
            mid2 = low + 2 * diff
            _, dist1 = optimize(mid1)
            _, dist2 = optimize(mid2)
            if dist1 < dist2:
                high = mid2
            elif dist1 > dist2:
                low = mid1
            else:
                low = mid1
                high = mid2
        res_x, r = optimize(low)
        res_y = low
        return [res_x, res_y, math.sqrt(r)]


class TestGeneral(unittest.TestCase):

    def test_tripart_search(self):
        tps = TriPartSearch()
        def fun1(x): return (x - 1) * (x - 1)
        assert abs(tps.find_floor(fun1, -5, 100) - 1) < 1e-5

        def fun2(x): return -(x - 1) * (x - 1)
        assert abs(tps.find_ceil(fun2, -5, 100) - 1) < 1e-5
        return

    def test_tripart_pack_tripart(self):
        tpt = TriPartPackTriPart()
        nodes = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        # 定义目标函数
        def target(x, y): return max([(x - p[0]) ** 2 + (y - p[1]) ** 2 for p in nodes])
        x0, y0, _ = tpt.find_floor(target, -10, 10, -10, 10)
        assert abs(x0) < 1e-5 and abs(y0) < 1e-5
        return


if __name__ == '__main__':
    unittest.main()
