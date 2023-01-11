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
P1462 通往奥格瑞玛的道路（https://www.luogu.com.cn/problem/P1462）使用带约束的最短路计算最终结果
L0787 K 站中转内最便宜的航班（https://leetcode.cn/problems/cheapest-flights-within-k-stops/）使用带约束的最短路计算最终结果

L2203 得到要求路径的最小带权子图（https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/）使用三个Dijkstra最短路获得结果
P1339 [USACO09OCT]Heat Wave G（https://www.luogu.com.cn/problem/P1339）标准最短路计算
P1342 请柬（https://www.luogu.com.cn/problem/P1342）正反两遍最短路
P1576 最小花费（https://www.luogu.com.cn/problem/P1576）堆优化转换成负数求最短路

P1821 [USACO07FEB] Cow Party S（https://www.luogu.com.cn/problem/P1821）正反两遍最短路
P1882 接力赛跑（https://www.luogu.com.cn/problem/P1882）转换为最短路求解最短路距离最远的点
P1907 设计道路（https://www.luogu.com.cn/problem/P1907）自定义建图计算最短路
P1744 采购特价商品（https://www.luogu.com.cn/problem/P1744）裸题最短路
P1529 [USACO2.4]回家 Bessie Come Home（https://www.luogu.com.cn/problem/P1529）裸题最短路
P1649 [USACO07OCT]Obstacle Course S（https://www.luogu.com.cn/problem/P1649）自定义距离计算的最短路
P2083 找人（https://www.luogu.com.cn/problem/P2083）反向最短路
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
        # Dijkstra求最短路，变成负数求可以求最长路（还是正权值）
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

    @staticmethod
    def gen_dijkstra_max_result(dct, src, dsc):

        # 求乘积最大的路
        inf = float("inf")
        dis = defaultdict(lambda: -inf)
        stack = [[-1, src]]
        dis[src] = 1
        while stack:
            d, i = heapq.heappop(stack)
            d = -d
            if dis[i] > d:
                continue
            for j in dct[i]:
                dj = dct[i][j] * d
                if dj > dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [-dj, j])
        return dis[dsc]


class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # 带约束的最短路 L0787
        dct = defaultdict(lambda: defaultdict(lambda: float("inf")))
        for u, v, p in flights:
            dct[u][v] = min(dct[u][v], p)

        stack = [[0, 0, src]]
        visit = defaultdict(lambda: float("inf"))
        while stack:
            cost, cnt, i = heapq.heappop(stack)
            if visit[i] <= cnt or cnt >= k + 2:
                continue
            if i == dst:
                return cost
            visit[i] = cnt
            for j in dct[i]:
                heapq.heappush(stack, [cost + dct[i][j], cnt + 1, j])
        return -1

    @staticmethod
    def main(n, cost, dct, s):
        # P1462
        def check():
            visit = [float("-inf")] * n
            stack = [[cost[0], 0, s]]
            visit[0] = s
            while stack:
                dis, i, bd = heapq.heappop(stack)
                if i == n - 1:
                    return dis
                if visit[i] > bd:
                    continue
                for j in dct[i]:
                    bj = bd - dct[i][j]
                    if bj >= 0 and visit[j] < bj:
                        visit[j] = bj
                        nex = dis if dis > cost[j] else cost[j]
                        heapq.heappush(stack, [nex, j, bj])
            return "AFK"
        return check()


class TestGeneral(unittest.TestCase):

    def test_dijkstra(self):
        djk = Dijkstra()
        dct = [{1: 1, 2: 4}, {2: 2}, {}]
        assert djk.get_dijkstra_result(dct, 0) == [0, 1, 3]
        return


if __name__ == '__main__':
    unittest.main()
