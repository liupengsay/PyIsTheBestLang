"""

"""

"""
算法：Dijkstra（单源最短路经算法）
功能：计算点到有向或者无向图里面其他点的最近距离
题目：

P3371 单源最短路径（弱化版）（https://www.luogu.com.cn/problem/P3371）
L2290 到达角落需要移除障碍物的最小数（https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/）计算最小代价
L2258 逃离火灾（https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）使用双源BFS计算等待时间后最短路求出路径上最小等待时间的最大值
P1629 邮递员送信（https://www.luogu.com.cn/problem/P1629）正反两个方向的最短路进行计算往返路程
P1462 通往奥格瑞玛的道路（https://www.luogu.com.cn/problem/P1462）使用二分加最短路计算最终结果
L2203 得到要求路径的最小带权子图（https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/）使用三个Dijkstra最短路获得结果


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
        dis[src] = 0
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j in dct[i]:
                dj = dct[i][j] + d
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j])
        return dis


class TestGeneral(unittest.TestCase):

    def test_dijkstra(self):
        djk = Dijkstra()
        dct = [{1: 1, 2: 4}, {2: 2}, {}]
        assert djk.get_dijkstra_result(dct, 0) == [0, 1, 3]
        return


if __name__ == '__main__':
    unittest.main()
