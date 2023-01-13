
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

"""
算法：计算几何、设计到平面坐标系上的一些问题求解
功能：xxx
题目：
P1665 正方形计数（https://www.luogu.com.cn/problem/P1665）枚举正方形对角线顶点计算可行个数
P2313 [HNOI2005]汤姆的游戏（https://www.luogu.com.cn/problem/P2313）判断点在矩形中或者圆形中
P2358 蚂蚁搬家（https://www.luogu.com.cn/problem/P2358）计算几何判断正方体上表面的点到下表面的点最短距离
参考：OI WiKi（xx）
"""


class Geometry:
    def __init__(self):
        return

    @staticmethod
    def compute_square_point(x0, y0, x2, y2):
        # 已知正方形对角线上的点，求另外两个点的坐标
        x1 = (x0 + x2 + y2 - y0) / 2
        y1 = (y0 + y2 + x0 - x2) / 2
        x3 = (x0 + x2 - y2 + y0) / 2
        y3 = (y0 + y2 - x0 + x2) / 2
        return (x1, y1), (x3, y3)

    @staticmethod
    def compute_square_area(x0, y0, x2, y2):
        # 已知正方形对角线上的点，求正方形面积，注意是整数
        ans = (x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)
        return ans // 2

    @staticmethod
    def compute_triangle_area(x1, y1, x2, y2, x3, y3):
        # 可用于判断点与三角形的位置关系
        return abs((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3)) / 2


class TestGeneral(unittest.TestCase):

    def test_geometry(self):
        gm = Geometry()
        assert gm.compute_square_point(1, 1, 6, 6) == ((6.0, 1.0), (1.0, 6.0))
        assert gm.compute_square_point(0, 0, 0, 2) == ((1.0, 1.0), (-1.0, 1.0))

        assert gm.compute_triangle_area(0, 0, 2, 0, 1, 1) == 1.0
        return


if __name__ == '__main__':
    unittest.main()
