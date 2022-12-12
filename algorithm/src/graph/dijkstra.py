"""
算法：Dijkstra（单源最短路经算法）
功能：计算点到有向或者无向图里面其他点的最近距离
题目：P3371 【模板】单源最短路径（弱化版）（https://www.luogu.com.cn/problem/P3371）
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


class Dijkstra:
    def __init__(self):
        return

    @staticmethod
    def get_dijkstra_result(dct, src):
        n = len(dct)
        dis = [float("inf")]*n
        stack = [[0, src]]
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] <= d:
                continue
            dis[i] = d
            for j in dct[i]:
                heapq.heappush(stack, [dct[i][j] + d, j])
        return dis


class TestGeneral(unittest.TestCase):

    def test_dijkstra(self):
        djk = Dijkstra()
        dct = [{1: 1, 2: 4}, {2: 2}, {}]
        assert djk.get_dijkstra_result(dct, 0) == [0, 1, 3]
        return


if __name__ == '__main__':
    unittest.main()
