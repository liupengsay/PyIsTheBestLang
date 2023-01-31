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
P1352 没有上司的舞会（https://www.luogu.com.cn/problem/P1352）树形DP，隔层进行动态规划转移
P1922 女仆咖啡厅桌游吧（https://www.luogu.com.cn/problem/P1922）树形DP，贪心进行子树与叶子节点的分配

P2016 战略游戏（https://www.luogu.com.cn/problem/P2016）树形DP瞭望每条边
968. 监控二叉树（https://leetcode.cn/problems/binary-tree-cameras/）树形DP监控每个节点

6294. 最大价值和与最小价值和的差值（https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/）树形换根DP，求去掉其中一个叶子节点的最大直径
124. 二叉树中的最大路径和（https://leetcode.cn/problems/binary-tree-maximum-path-sum/）树形DP
P1122 最大子树和（https://www.luogu.com.cn/problem/P1122）计算最大的连通块和
F - Expensive Expense （https://atcoder.jp/contests/abc222/tasks/abc222_f）换根DP

P2932 [USACO09JAN]Earthquake Damage G（https://www.luogu.com.cn/problem/P2932）树形DP统计子树个数与贪心安排最小损坏个数
P2996 [USACO10NOV]Visiting Cows G（https://www.luogu.com.cn/problem/P2996）树形DP

P3074 [USACO13FEB]Milk Scheduling S（https://www.luogu.com.cn/problem/P3074）树的最长路径（广搜DP记录最长时间也可以）
P3884 [JLOI2009]二叉树问题（https://www.luogu.com.cn/problem/P3884）基础树形DP计算两点间路径变种长度
P3915 树的分解（https://www.luogu.com.cn/problem/P3915）递归拆解生成等大小的连通块
P4615 [COCI2017-2018#5] Birokracija（https://www.luogu.com.cn/problem/P4615）树形DP
P5002 专心OI - 找祖先（https://www.luogu.com.cn/problem/P5002）使用树形DP与容斥原理进行计数

P5651 基础最短路练习题（https://www.luogu.com.cn/problem/P5651）脑筋急转弯使用并查集去环，转换为树形DP里面任意两点路径的异或和

P6591 [YsOI2020]植树（https://www.luogu.com.cn/problem/P6591）换根DP，即无根树递归判断每个节点作为根节点的情况


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
    def change_root_dp(n: int, edges: List[List[int]], price: List[int]):
        # 模板： 换根DP
        edge = [[] for _ in range(n)]
        for u, v in edges:
            edge[u].append(v)
            edge[v].append(u)

        @lru_cache(None)
        def dfs(i, fa):
            # 注意在星状图的复杂度是O(n^2)（还有一种特殊的树结构是树链）
            # 也是求以此为根的最大路径
            ans = 0
            for j in edge[i]:
                if j != fa:
                    cur = dfs(j, i)
                    ans = ans if ans > cur else cur
            return ans + price[i]

        return max(dfs(i, -1) - price[i] for i in range(n))

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

        son_dis = [0] * n
        dfs(0)

        def dfs(x):
            for y in tree[x]:
                father_dis[y] = (
                    son_dis[x] - son_dis[y] - son_count[y]) + father_dis[x] + n - son_count[y]
                dfs(y)
            return

        father_dis = [0] * n
        dfs(0)
        return [father_dis[i] + son_dis[i] for i in range(n)]

    @staticmethod
    def longest_path_through_node(dct):

        # 模板：换根DP，两遍DFS获取从下往上与从上往下的DP信息
        n = len(dct)

        # 两遍DFS获取从下往上与从上往下的节点最远距离
        def dfs(x, fa):
            res = [0, 0]
            for y in dct[x]:
                if y != fa:
                    dfs(y, x)
                    res.append(max(down_to_up[y]) + 1)
            down_to_up[x] = nlargest(2, res)
            return

        # 默认以 0 为根
        down_to_up = [[] for _ in range(n)]
        dfs(0, -1)

        def dfs(x, pre, fa):
            up_to_down[x] = pre
            son = [0, 0]
            for y in dct[x]:
                if y != fa:
                    son.append(max(down_to_up[y]))
            son = nlargest(2, son)

            for y in dct[x]:
                if y != fa:
                    father = pre + 1
                    tmp = son[:]
                    if max(down_to_up[y]) in tmp:
                        tmp.remove(max(down_to_up[y]))
                    if tmp[0]:
                        father = father if father > tmp[0] + 2 else tmp[0] + 2
                    dfs(y, father, x)
            return

        up_to_down = [0] * n
        # 默认以 0 为根
        dfs(0, 0, -1)
        # 树的直径、核心可通过这两个数组计算得到，其余类似的递归可参照这种方式
        return up_to_down, down_to_up


class TreeDiameter:
    # 任取树中的一个节点x，找出距离它最远的点y，那么点y就是这棵树中一条直径的一个端点。我们再从y出发，找出距离y最远的点就找到了一条直径。
    # 这个算法依赖于一个性质：对于树中的任一个点，距离它最远的点一定是树上一条直径的一个端点。
    def __init__(self, edge):
        self.edge = edge
        self.n = len(self.edge)
        return

    def get_farest(self, node):
        q = deque([(node, -1)])
        while q:
            node, pre = q.popleft()
            for x in self.edge[node]:
                if x != pre:
                    q.append((x, node))
        return node

    def get_diameter_node(self):
        # 获取树的直径端点
        x = self.get_farest(0)
        y = self.get_farest(x)
        return x, y


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
