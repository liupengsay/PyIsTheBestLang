"""

"""

"""
算法：拓扑排序、内向基环树（有向或者无向，连通块有k个节点以及k条边）
功能：有向图进行排序，无向图在选定根节点的情况下也可以进行拓扑排序
题目：xx（xx）

L2392 给定条件下构造矩阵（https://leetcode.cn/problems/build-a-matrix-with-conditions/）分别通过行列的拓扑排序来确定数字所在索引
L2127 参加会议的最多员工数（https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/）拓扑排序确定内向基环，按照环的大小进行贪心枚举
（无向图的内向基环树，求简单路径的树枝连通）F - Well-defined Path Queries on a Namori：https://atcoder.jp/contests/abc266/submissions/me
内向基环树介绍：https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/solution/nei-xiang-ji-huan-shu-tuo-bu-pai-xu-fen-c1i1b/
2360. 图中的最长环（https://leetcode.cn/problems/longest-cycle-in-a-graph/）
2127. 参加会议的最多员工数（https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/）
P1960 郁闷的记者（https://www.luogu.com.cn/problem/P1960）计算拓扑排序是否唯一
P1992 不想兜圈的老爷爷（https://www.luogu.com.cn/problem/P1992）拓扑排序计算有向图是否有环
269. 火星词典（https://leetcode.cn/problems/alien-dictionary/）经典按照字典序建图，与拓扑排序的应用
P2712 摄像头（https://www.luogu.com.cn/problem/P2712）拓扑排序计算非环节点数
P6145 [USACO20FEB]Timeline G（https://www.luogu.com.cn/problem/P6145）经典拓扑排序计算每个节点最晚的访问时间点

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


class TopologicalSort:
    def __init__(self):
        return

    @staticmethod
    def get_rank(n, edges):
        dct = [list() for _ in range(n)]
        degree = [0]*n
        for i, j in edges:
            degree[j] += 1
            dct[i].append(j)
        stack = [i for i in range(n) if not degree[i]]
        visit = [-1]*n
        step = 0
        while stack:
            for i in stack:
                visit[i] = step
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
            step += 1
        return visit

    # 内向基环树写法 https://atcoder.jp/contests/abc266/submissions/37717739
    # def main(ac=FastIO()):
    #     n = ac.read_int()
    #     edge = [[] for _ in range(n)]
    #     uf = UnionFind(n)
    #
    #     degree = [0] * n
    #     for _ in range(n):
    #         u, v = ac.read_ints_minus_one()
    #         edge[u].append(v)
    #         edge[v].append(u)
    #         degree[u] += 1
    #         degree[v] += 1
    #
    #     que = deque()
    #     for i in range(n):
    #         if degree[i] == 1:
    #             que.append(i)
    #     while que:
    #         now = que.popleft()
    #         nex = edge[now][0]
    #         degree[now] -= 1
    #         degree[nex] -= 1
    #         edge[nex].remove(now)
    #         uf.union(now, nex)
    #         if degree[nex] == 1:
    #             que.append(nex)
    #
    #     q = ac.read_int()
    #     for _ in range(q):
    #         x, y = ac.read_ints_minus_one()
    #         if uf.is_connected(x, y):
    #             ac.st("Yes")
    #         else:
    #             ac.st("No")
    #     return

    @staticmethod
    def is_topology_unique(dct, degree, n):

        # 保证存在拓扑排序的情况下判断是否唯一
        ans = []
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            ans.extend(stack)
            if len(stack) > 1:
                return False
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return True

    @staticmethod
    def is_topology_loop(edge, degree, n):

        # 使用拓扑排序判断有向图是否存在环
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return sum(degree) == 0


class TestGeneral(unittest.TestCase):

    def test_topological_sort(self):
        ts = TopologicalSort()
        n = 5
        edges = [[0, 1], [0, 2], [1, 4], [2, 3], [3, 4]]
        assert ts.get_rank(n, edges) == [0, 1, 1, 2, 3]
        return


if __name__ == '__main__':
    unittest.main()
