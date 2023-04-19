"""

"""
from algorithm.src.fast_io import FastIO

"""
算法：凸包、最小圆覆盖
功能：求点集的子集组成最小凸包上
题目：

===================================力扣===================================
1924 安装栅栏 II（https://leetcode.cn/problems/erect-the-fence-ii/）求出最小凸包后使用三分套三分求解最小圆覆盖，随机增量法求最小圆覆盖

===================================洛谷===================================
P1742 最小圆覆盖（https://www.luogu.com.cn/problem/P1742）随机增量法求最小圆覆盖
P3517 [POI2011]WYK-Plot（https://www.luogu.com.cn/problem/P3517）二分套二分，随机增量法求最小圆覆盖

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


class MinCircleOverlap:
    def __init__(self):
        self.pi = math.acos(-1)
        self.esp = 10 ** (-10)
        return

    def get_min_circle_overlap(self, points: List[List[int]]):
        # 模板：随机增量法求解最小圆覆盖

        def cross(a, b):
            return a[0] * b[1] - b[0] * a[1]

        def intersection_point(p1, v1, p2, v2):
            # 求解两条直线的交点
            u = (p1[0] - p2[0], p1[1] - p2[1])
            t = cross(v2, u) / cross(v1, v2)
            return p1[0] + v1[0] * t, p1[1] + v1[1] * t

        def is_point_in_circle(circle_x, circle_y, circle_r, x, y):
            res = math.sqrt((x - circle_x) ** 2 + (y - circle_y) ** 2)
            if abs(res - circle_r) < self.esp:
                return True
            if res < circle_r:
                return True
            return False

        def vec_rotate(v, theta):
            x, y = v
            return x * math.cos(theta) + y * math.sin(theta), -x * math.sin(theta) + y * math.cos(theta)

        def get_out_circle(x1, y1, x2, y2, x3, y3):
            xx1, yy1 = (x1 + x2) / 2, (y1 + y2) / 2
            vv1 = vec_rotate((x2 - x1, y2 - y1), self.pi / 2)
            xx2, yy2 = (x1 + x3) / 2, (y1 + y3) / 2
            vv2 = vec_rotate((x3 - x1, y3 - y1), self.pi / 2)
            pp = intersection_point((xx1, yy1), vv1, (xx2, yy2), vv2)
            res = math.sqrt((pp[0] - x1) ** 2 + (pp[1] - y1) ** 2)
            return pp[0], pp[1], res

        random.shuffle(points)
        n = len(points)
        p = points

        # 圆心与半径
        cc1 = (p[0][0], p[0][1], 0)
        for ii in range(1, n):
            if not is_point_in_circle(cc1[0], cc1[1], cc1[2], p[ii][0], p[ii][1]):
                cc2 = (p[ii][0], p[ii][1], 0)
                for jj in range(ii):
                    if not is_point_in_circle(cc2[0], cc2[1], cc2[2], p[jj][0], p[jj][1]):
                        dis = math.sqrt((p[jj][0] - p[ii][0]) ** 2 + (p[jj][1] - p[ii][1]) ** 2)
                        cc3 = ((p[jj][0] + p[ii][0]) / 2, (p[jj][1] + p[ii][1]) / 2, dis / 2)
                        for kk in range(jj):
                            if not is_point_in_circle(cc3[0], cc3[1], cc3[2], p[kk][0], p[kk][1]):
                                cc3 = get_out_circle(p[ii][0], p[ii][1], p[jj][0], p[jj][1], p[kk][0], p[kk][1])
                        cc2 = cc3
                cc1 = cc2

        return cc1


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1924(trees: List[List[int]]) -> List[float]:
        # 模板：随机增量法求最小圆覆盖
        ans = MinCircleOverlap().get_min_circle_overlap(trees)
        return list(ans)

    @staticmethod
    def lg_p1742(ac=FastIO()):
        # 模板：随机增量法求最小圆覆盖
        n = ac.read_int()
        nums = [ac.read_list_floats() for _ in range(n)]
        x, y, r = MinCircleOverlap().get_min_circle_overlap(nums)
        ac.st(r)
        ac.lst([x, y])
        return

    @staticmethod
    def lg_3517(ac=FastIO()):

        # 模板：随机增量法求最小圆覆盖
        n, m = ac.read_ints()
        nums = [ac.read_list_ints() for _ in range(n)]

        def check(r):

            def circle(lst):
                x, y, rr = MinCircleOverlap().get_min_circle_overlap(lst)
                return x, y, rr

            cnt = i = 0
            res = []
            while i < n:
                left = i
                right = n-1
                while left < right-1:
                    mm = left+(right-left)//2
                    if circle(nums[i:mm+1])[2] <= r:
                        left = mm
                    else:
                        right = mm
                ll = circle(nums[i:right+1])
                if ll[2] > r:
                    ll = circle(nums[i:left+1])
                    i = left+1
                else:
                    i = right + 1
                res.append(ll[:-1])
                cnt += 1
            return res, cnt <= m

        low = 0
        high = 4*10**6
        error = 10**(-6)
        while low < high-error:
            mid = low+(high-low)/2
            if check(mid)[1]:
                high = mid
            else:
                low = mid

        nodes, flag = check(low)
        rrr = low
        if not flag:
            nodes, flag = check(high)
            rrr = high
        ac.st(rrr)
        ac.st(len(nodes))
        for a in nodes:
            ac.lst([round(a[0], 10), round(a[1], 10)])
        return


class TestGeneral(unittest.TestCase):

    def test_convex_hull(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
