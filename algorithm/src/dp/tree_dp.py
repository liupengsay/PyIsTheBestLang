"""

"""
"""
算法：树形DP
功能：在树形或者图结构上进行DP，有换根DP，自顶向下和自底向上DP
题目：

L2458 移除子树后的二叉树高度（https://leetcode.cn/problems/height-of-binary-tree-after-subtree-removal-queries/）跑两边DFS进行自顶向下和自底向上DP结合

L2440 创建价值相同的连通块（https://leetcode.cn/problems/create-components-with-same-value/）利用总和的因子和树形递归判断连通块是否可行
L1569 将子数组重新排序得到同一个二叉查找树的方案数（https://leetcode.cn/problems/number-of-ways-to-reorder-array-to-get-same-bst/solution/by-liupengsay-yi3h/）
P1395 会议（https://leetcode.cn/problems/create-components-with-same-value/）树的总距离，单个节点距离其他所有节点的最大距离

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
from heapq import nlargest


class TreeDP:
    def __init__(self):
        return


    @staticmethod
    def sum_of_distances_in_tree(n: int, edges):
        # 计算节点到所有其他节点的总距离
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        tree = [[] for _ in range(n)]
        stack = [0]
        visit = {0}
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    if j not in visit:
                        visit.add(j)
                        nex.append(j)
                        tree[i].append(j)
            stack = nex[:]

        def dfs(x):
            res = 1
            for y in tree[x]:
                res += dfs(y)
            son_count[x] = res
            return res

        son_count = [0] * n
        dfs(0)

        def dfs(x):
            res = son_count[x] - 1
            for y in tree[x]:
                res += dfs(y)
            son_dis[x] = res
            return res

        son_dis = [0]*n
        dfs(0)

        def dfs(x):
            for y in tree[x]:
                father_dis[y] = (son_dis[x] - son_dis[y] - son_count[y]) + father_dis[x] + n-son_count[y]
                dfs(y)
            return

        father_dis = [0]*n
        dfs(0)
        return [father_dis[i]+son_dis[i] for i in range(n)]


    @staticmethod
    def longest_path_through_node(dct):
        n = len(dct)

        # 两遍DFS获取从下往上与从上往下的节点最远距离
        def dfs(x):
            visit[x] = 1
            res = [0, 0]
            for y in dct[x]:
                if not visit[y]:
                    dfs(y)
                    res.append(max(down_to_up[y])+1)
            down_to_up[x] = nlargest(2, res)
            return

        # 默认以 0 为根
        visit = [0]*n
        down_to_up = [[] for _ in range(n)]
        dfs(0)

        def dfs(x, pre):
            visit[x] = 1
            up_to_down[x] = pre
            son = [0, 0]
            for y in dct[x]:
                if not visit[y]:
                    son.append(max(down_to_up[y]))
            son = nlargest(2, son)

            for y in dct[x]:
                if not visit[y]:
                    father = pre + 1
                    tmp = son[:]
                    if max(down_to_up[y]) in tmp:
                        tmp.remove(max(down_to_up[y]))
                    if tmp[0]:
                        father = father if father > tmp[0]+2 else tmp[0]+2
                    dfs(y, father)
            return

        visit = [0]*n
        up_to_down = [0] * n
        # 默认以 0 为根
        dfs(0, 0)
        # 树的直径、核心可通过这两个数组计算得到，其余类似的递归可参照这种方式
        return up_to_down, down_to_up


class TestGeneral(unittest.TestCase):

    def test_tree_dp(self):
        td = TreeDP()
        n = 5
        edges = [[0, 1], [0, 2], [2, 4], [1, 3]]
        assert td.sum_of_distances_in_tree(n, edges) == [6, 7, 7, 10, 10]

        dct = [[1, 2], [0, 3], [0, 4], [1], [2]]
        up_to_down, down_to_up = td.longest_path_through_node(dct)
        assert up_to_down == [0, 3, 3, 4, 4]
        assert down_to_up == [[2, 2], [1, 0], [1, 0], [0, 0], [0, 0]]
        return


if __name__ == '__main__':
    unittest.main()
