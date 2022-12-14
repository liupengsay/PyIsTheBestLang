"""
算法：最小生成树（Kruskal算法和Prim算法两种）
功能：计算无向图边权值和最小的生成树
题目：
P3366 最小生成树（https://www.luogu.com.cn/problem/P3366）计算最小生成树的权值和
L1489 找到最小生成树里的关键边和伪关键边（https://leetcode.cn/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/）计算最小生成树的关键边与伪关键边
Prim在稠密图中比Kruskal优，在稀疏图中比Kruskal劣。Prim是以更新过的节点的连边找最小值，Kruskal是直接将边排序。
两者其实都是运用贪心的思路
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

class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n

    def find(self, x):
        if x != self.root[x]:
            # 在查询的时候合并到顺带直接根节点
            root_x = self.find(self.root[x])
            self.root[x] = root_x
            return root_x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] >= self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # 将非根节点的秩赋0
        self.size[root_x] = 0
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        # 获取每个根节点对应的组
        part = defaultdict(list)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        # 获取每个根节点对应的组大小
        size = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
        return size


class MininumSpanningTree:
    def __init__(self, edges, n):
        # n个节点
        self.n = n
        # m条权值边edges
        self.edges = edges
        self.cost = 0
        self.gen_minimum_spanning_tree()
        return

    def gen_minimum_spanning_tree(self):
        self.edges.sort(key=lambda item: item[2])
        # 贪心按照权值选择边进行连通合并
        uf = UnionFind(self.n)
        for x, y, z in self.edges:
            if uf.union(x, y):
                self.cost += z
        # 不能形成生成树
        if uf.part != 1:
            self.cost = -1
        return


class TestGeneral(unittest.TestCase):

    def test_minimum_spanning_tree(self):
        n = 3
        edges = [[0, 1, 2], [1, 2, 3], [2, 0, 4]]
        mst = MininumSpanningTree(edges, n)
        assert mst.cost == 5

        n = 4
        edges = [[0, 1, 2], [1, 2, 3], [2, 0, 4]]
        mst = MininumSpanningTree(edges, n)
        assert mst.cost == -1
        return


if __name__ == '__main__':
    unittest.main()