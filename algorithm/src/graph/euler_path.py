"""
算法：欧拉路径（使用深度优先搜索里面的Hierholzer算法）
功能：求解有向图与无向图中的欧拉路径，定义比较复杂且不统一，须根据实际情况作适配与调整

题目：P7771 【模板】欧拉路径（https://www.luogu.com.cn/problem/P7771）
753. 破解保险箱（https://leetcode.cn/problems/cracking-the-safe/solution/er-xu-cheng-ming-jiu-xu-zui-by-liupengsa-lm77/）

参考：OI WiKi（https://oi-wiki.org/graph/euler/）

https://www.jianshu.com/p/8394b8e5b878
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


class EulerPath:
    def __init__(self, pairs):
        # 数组形式存储的有向连接关系
        self.pairs = pairs
        # 欧拉路径上的每条边和经过的几点
        self.paths = list()
        self.nodes = list()

        self.get_euler_path()
        return

    def get_euler_path(self):
        # 存顶点的出入度
        degree = defaultdict(lambda: [0, 0])
        # 存储图关系
        edge = defaultdict(list)
        for i, j in self.pairs:
            degree[i][0] += 1
            degree[j][1] += 1
            edge[i].append(j)

        # 根据字典序优先访问较小的
        for i in edge:
            edge[i].sort(reverse=True)

        # 寻找起始节点
        start = self.pairs[0][0]
        for i in degree:
            # 如果有节点出度比入度恰好多 1，那么只有它才能是起始节点，且应该只有一个，否则任选一个开始
            if degree[i][0] - degree[i][1] == 1:
                start = i
                break

        def dfs(pre):
            # 使用深度优先搜索（Hierholzer算法）求解欧拉通路
            while edge[pre]:
                nex = edge[pre].pop()
                dfs(nex)
                self.nodes.append(nex)
                self.paths.append([pre, nex])
            return

        dfs(start)
        # 注意判断所有边都经过的才算欧拉路径
        self.paths.reverse()
        self.nodes.append(start)
        self.nodes.reverse()
        return


class TestGeneral(unittest.TestCase):

    def test_euler_path(self):
        pairs = [[1, 2], [2, 3], [3, 4], [4, 3], [3, 2], [2, 1]]
        ep = EulerPath(pairs)
        assert ep.paths == [[1, 2], [2, 3], [3, 4], [4, 3], [3, 2], [2, 1]]

        pairs = [[1, 3], [2, 1], [4, 2], [3, 3], [1, 2], [3, 4]]
        ep = EulerPath(pairs)
        assert ep.nodes == [1, 2, 1, 3, 3, 4, 2]
        return


if __name__ == '__main__':
    unittest.main()
