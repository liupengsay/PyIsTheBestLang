
"""

"""

"""
算法：Floyd（单源最短路经算法）
功能：计算点到有向或者无向图里面其他点的最近距离
题目：
P1119 灾后重建 （https://www.luogu.com.cn/problem/P1119）离线查询加Floyd动态更新经过中转站的起终点距离

参考：OI WiKi（xx）
"""


# class Solution:
#     def shortestPathLength(self, graph: List[List[int]]):
#         n = len(graph)
#         d = [[n + 1] * n for _ in range(n)]
#         for i in range(n):
#             for j in graph[i]:
#                 d[i][j] = 1
#
#         # 使用 floyd 算法预处理出所有点对之间的最短路径长度
#         for k in range(n):
#             for i in range(n):
#                 for j in range(n):
#                     d[i][j] = min(d[i][j], d[i][k] + d[k][j])
#         return dp




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
class Luogu:
    def __init__(self):
        return

    @staticmethod
    def main_p1119(n, repair, edges, queries):

        # 设置初始值距离
        dis = [[float("inf")] * n for _ in range(n)]
        for a, b, c in edges:
            dis[a][b] = dis[b][a] = c
        for i in range(n):
            dis[i][i] = 0

        res = []
        # 修复村庄之后用 Floyd算法 更新以该村庄为中转的距离
        k = 0
        for x, y, t in queries:
            while k < n and repair[k] <= t:
                for a in range(n):
                    for b in range(a + 1, n):
                        cur = dis[a][k] + dis[k][b]
                        if dis[a][b] > cur:
                            dis[a][b] = dis[b][a] = cur
                k += 1
            if dis[x][y] < float("inf") and x < k and y < k:
                res.append(dis[x][y])
            else:
                res.append(-1)
        return res


class TestGeneral(unittest.TestCase):

    def test_luogu(self):
        luogu = Luogu()
        n = 4
        repair = [1, 2, 3, 4]
        edges = [[0, 2, 1], [2, 3, 1], [3, 1, 2], [2, 1, 4], [0, 3, 5]]
        queries = [[2, 0, 2], [0, 1, 2], [0, 1, 3], [0, 1, 4]]
        assert luogu.main_p1119(n, repair, edges, queries) == [-1, -1, 5, 4]
        return


if __name__ == '__main__':
    unittest.main()
