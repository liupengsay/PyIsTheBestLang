"""
算法：支配树
功能：支配树算法是一种用来求解图中支配性的算法。它通常用于寻找图中的支配点（即点的集合，它们可以支配图中的所有点）。下面是一个简单的支配树算法的模板，它使用 Python 语言实现：
题目：P5180 【模板】支配树（https://www.luogu.com.cn/problem/P5180）
参考：OI WiKi（）
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


# 定义一个支配树类
class DominatingTree:
    def __init__(self, n):
        # 初始化支配树，n 为图中点的数量
        self.n = n
        self.edges = []  # 存储图中的边
        self.dominators = [-1] * n  # 存储每个点的支配点

    def add_edge(self, u, v):
        # 添加一条从点 u 到点 v 的边
        self.edges.append((u, v))

    def build(self):
        # 构建支配树
        # 使用并查集维护每个点的支配点
        parent = [i for i in range(self.n)]

        # 遍历图中的每条边，更新支配点
        for (u, v) in self.edges:
            pu = self.find(parent, u)
            pv = self.find(parent, v)
            if pu != pv:
                parent[pv] = pu

        # 所有节点的支配点都是根节点
        for i in range(self.n):
            self.dominators[i] = self.find(parent, i)

    def find(self, parent, i):
        # 并查集的 find 操作
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    def get_dominators(self):
        # 返回每个点的支配点
        return self.dominators



class TestGeneral(unittest.TestCase):
    def test_euler_phi(self):
        # 创建支配树对象
        dt = DominatingTree(5)

        # 添加图中的边
        dt.add_edge(0, 1)
        dt.add_edge(1, 2)
        dt.add_edge(2, 3)
        dt.add_edge(3, 4)

        # 构建支配树
        dt.build()

        # 获取每个点的支配点
        dominators = dt.get_dominators()

        # 输出每个点的支配点
        for i in range(5):
            print(f"点 {i} 的支配点为 {dominators[i]}")
        return


if __name__ == '__main__':
    unittest.main()
