
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
算法：Prufer序列
功能：Prufer 序列 (Prufer code)，这是一种将带标号的无根树用一个唯一的整数序列表示的方法，可以生成带标号无根树与prufer数列的一种双射关系。
题目：

===================================洛谷===================================
P6086 【模板】Prufer 序列（https://www.luogu.com.cn/problem/P6086）Prufer 序列裸题
P2817 宋荣子的城堡（https://www.luogu.com.cn/problem/P2817）Cayley公式计算方案数

参考：OI WiKi（https://oi-wiki.org/graph/prufer/）
"""


class PruferAndTree:
    def __init__(self):
        """默认以0为最小标号"""
        return

    @staticmethod
    def adj_to_parent(adj, root):

        def dfs(v):
            for u in adj[v]:
                if u != parent[v]:
                    parent[u] = v
                    dfs(u)

        n = len(adj)
        parent = [-1] * n
        dfs(root)
        return parent

    @staticmethod
    def parent_to_adj(parent):
        n = len(parent)
        adj = [[] for _ in range(n)]
        for i in range(n):
            if parent[i] != -1:  # 即 i!=root
                adj[i].append(parent[i])
                adj[parent[i]].append(i)
        return parent

    def tree_to_prufer(self, adj, root):
        # 以root为根的带标号树生成prufer序列，adj为邻接关系
        parent = self.adj_to_parent(adj, root)
        n = len(adj)
        # 统计度数，以较小的叶子节点序号开始
        ptr = -1
        degree = [0] * n
        for i in range(0, n):
            degree[i] = len(adj[i])
            if degree[i] == 1 and ptr == -1:
                ptr = i

        # 生成prufer序列
        code = [0] * (n - 2)
        leaf = ptr
        for i in range(0, n - 2):
            nex = parent[leaf]
            code[i] = nex
            degree[nex] -= 1
            if degree[nex] == 1 and nex < ptr:
                leaf = nex
            else:
                ptr = ptr + 1
                while degree[ptr] != 1:
                    ptr = ptr + 1
                leaf = ptr
        return code

    @staticmethod
    def prufer_to_tree(code, root):
        # prufer序列生成以root为根的带标号树
        n = len(code) + 2

        # 根据度确定初始叶节点
        degree = [1]*n
        for i in code:
            degree[i] += 1
        ptr = 0
        while degree[ptr] != 1:
            ptr += 1
        leaf = ptr

        # 逆向工程进行还原
        adj = [[] for _ in range(n)]
        for v in code:
            adj[v].append(leaf)
            adj[leaf].append(v)
            degree[v] -= 1
            if degree[v] == 1 and v < ptr and v != root:
                leaf = v
            else:
                ptr += 1
                while degree[ptr] != 1:
                    ptr += 1
                leaf = ptr

        # 最后还由就是生成prufer序列剩下的根和叶子节点
        adj[leaf].append(root)
        adj[root].append(leaf)
        for i in range(n):
            adj[i].sort()
        return adj


class TestGeneral(unittest.TestCase):
    def test_tree_to_prufer(self):
        ptt = PruferAndTree()
        adj = [[1, 2, 3, 4], [0], [0, 5, 6], [0], [0], [2], [2]]
        code = [0, 0, 0, 2, 2]
        assert ptt.tree_to_prufer(adj, root=6) == code

        ptt = PruferAndTree()
        adj = [[1], [0, 2, 3, 6], [1, 4, 5], [1], [2], [2], [1]]
        code = [1, 1, 2, 2, 1]
        assert ptt.tree_to_prufer(adj, root=1) == code
        return

    def test_prufer_to_tree(self):
        ptt = PruferAndTree()
        code = [0, 0, 0, 2, 2]
        adj = [[1, 2, 3, 4], [0], [0, 5, 6], [0], [0], [2], [2]]
        assert ptt.prufer_to_tree(code, root=6) == adj

        ptt = PruferAndTree()
        code = [1, 1, 2, 2, 1]
        adj = [[1], [0, 2, 3, 6], [1, 4, 5], [1], [2], [2], [1]]
        assert ptt.prufer_to_tree(code, root=1) == adj
        return


if __name__ == '__main__':
    unittest.main()
