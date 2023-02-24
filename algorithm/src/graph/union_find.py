"""

"""

"""
算法：并查集、可持久化并查集
功能：用来处理图论相关的联通问题，通常结合逆向思考、置换环或者离线查询进行求解，连通块不一定是秩大小，也可以是最大最小值、和等
题目：

===================================力扣===================================
1697. 检查边长度限制的路径是否存在（https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths/）排序后离线查询两点间所有路径的最大边权值
2503. 矩阵查询可获得的最大分数（https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths/）排序后离线查询与起点相连的连通块的大小
2421. 好路径的数目（https://leetcode.cn/problems/number-of-good-paths/）根据权值进行排序更新并查集计算连通分块满足条件的节点对数
2382. 删除操作后的最大子段和（https://leetcode.cn/problems/maximum-segment-sum-after-removals/）逆向进行访问查询并更新连通块的结果
2334. 元素值大于变化阈值的子数组（https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/）排序后枚举动态维护并查集连通块
2158. 每天绘制新区域的数量（https://leetcode.cn/problems/amount-of-new-area-painted-each-day/）使用并查集维护区间左端点，不断进行合并
2157. 字符串分组（https://leetcode.cn/problems/groups-of-strings/）利用字母的有限数量进行变换枚举分组
2076. 处理含限制条件的好友请求（https://leetcode.cn/problems/process-restricted-friend-requests/）使用并查集变种，维护群体的不喜欢关系


===================================洛谷===================================
P3367 并查集（https://www.luogu.com.cn/problem/P3367）计算连通分块的数量
P5836 Milk Visits S（https://www.luogu.com.cn/problem/P5836）使用两个并查集进行不同方面的查询
P3144 [USACO16OPEN]Closing the Farm S（https://www.luogu.com.cn/problem/P3144）逆序并查集，考察连通块的数量
P5836 [USACO19DEC]Milk Visits S（https://www.luogu.com.cn/problem/P5836）两个并查集进行连通情况查询
P5877 棋盘游戏（https://www.luogu.com.cn/problem/P5877）正向模拟实时计算更新连通块的数量
P6111 [USACO18JAN]MooTube S（https://www.luogu.com.cn/problem/P6111）并查集加离线查询进行计算
P6121 [USACO16OPEN]Closing the Farm G（https://www.luogu.com.cn/problem/P6121）逆序并查集根据连通块大小进行连通性判定
P6153 询问（https://www.luogu.com.cn/problem/P6153）经典并查集思想贪心题，体现了并查集的思想

================================CodeForces================================
D. Roads not only in Berland（https://codeforces.com/problemset/problem/25/D）并查集将原来的边断掉重新来连接使得成为一整个连通集

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


# 标准并查集
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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1697(n: int, edge_list: List[List[int]], queries: List[List[int]]) -> List[bool]:
        # 模板：并查集与离线排序查询结合
        m = len(queries)

        # 按照 limit 排序
        ind = list(range(m))
        ind.sort(key=lambda x: queries[x][2])

        # 按照边权值排序
        edge_list.sort(key=lambda x: x[2])
        uf = UnionFind(n)
        i = 0
        k = len(edge_list)
        ans = []
        # 查询 queries 里面的 [p, q, limit] 即 p 和 q 之间存在最大边权值严格小于 limit 的路径是否成立
        for j in ind:
            # 实时加入可用于连通的边并查询结果
            p, q, limit = queries[j]
            while i < k and edge_list[i][2] < limit:
                uf.union(edge_list[i][0], edge_list[i][1])
                i += 1
            ans.append([j, uf.is_connected(p, q)])

        # 按照顺序返回结果
        ans.sort(key=lambda x: x[0])
        return [an[1] for an in ans]

    @staticmethod
    def lc_2503(grid: List[List[int]], queries: List[int]) -> List[int]:
        # 模板：并查集与离线排序查询结合
        dct = []
        # 根据邻居关系进行建图处理
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if i + 1 < m:
                    x, y = grid[i][j], grid[i + 1][j]
                    dct.append([i * n + j, i * n + n + j, x if x > y else y])
                if j + 1 < n:
                    x, y = grid[i][j], grid[i][j + 1]
                    dct.append([i * n + j, i * n + 1 + j, x if x > y else y])
        dct.sort(key=lambda d: d[2])
        uf = UnionFind(m * n)

        # 按照查询值的大小排序，依次进行查询
        k = len(queries)
        ind = list(range(k))
        ind.sort(key=lambda d: queries[d])

        # 根据查询值的大小利用指针持续更新并查集
        ans = [0]*k
        j = 0
        length = len(dct)
        for i in ind:
            cur = queries[i]
            while j < length and dct[j][2] < cur:
                uf.union(dct[j][0], dct[j][1])
                j += 1
            if cur > grid[0][0]:
                ans[i] = uf.size[uf.find(0)]
        return ans

    @staticmethod
    def lc_2421(vals: List[int], edges: List[List[int]]) -> int:
        # 模板：并查集与离线排序查询结合
        n = len(vals)
        index = defaultdict(list)
        for i in range(n):
            index[vals[i]].append(i)
        edges.sort(key=lambda x: max(vals[x[0]], vals[x[1]]))
        uf = UnionFind(n)
        # 离线查询计数
        i = 0
        m = len(edges)
        ans = 0
        for val in sorted(index):
            while i < m and vals[edges[i][0]] <= val and vals[edges[i][1]] <= val:
                uf.union(edges[i][0], edges[i][1])
                i += 1
            cnt = Counter(uf.find(x) for x in index[val])
            for w in cnt.values():
                ans += w * (w - 1) // 2 + w
        return ans


# 可持久化并查集
class PersistentUnionFind:
    def __init__(self, n):
        self.rank = [0]*n
        self.root = list(range(n))
        self.version = [float("inf")]*n

    def union(self, x, y, tm):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.version[root_y] = tm
                self.root[root_y] = root_x
            else:
                self.version[root_x] = tm
                self.root[root_x] = root_y
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1
            return True
        return False

    def find(self, x, tm=float("inf")):
        if x == self.root[x] or self.version[x] >= tm:
            return x
        return self.find(self.root[x], tm)

    def is_connected(self, x, y, tm):
        return self.find(x, tm) == self.find(y, tm)


class TestGeneral(unittest.TestCase):

    def test_union_find(self):
        uf = UnionFind(5)
        for i, j in [[0, 1], [1, 2]]:
            uf.union(i, j)
        assert uf.part == 3
        return

    def test_solutioon(self):
        # 离线根据时间戳排序进行查询
        sl = Solution()
        n = 3
        edge_list = [[0, 1, 2], [1, 2, 4], [2, 0, 8], [1, 0, 16]]
        queries = [[0, 1, 2], [0, 2, 5]]
        assert sl.distance_limited_paths_exist(n, edge_list, queries) == [False, True]

    def test_persistent_union_find(self):
        # 在线根据历史版本时间戳查询
        n = 3
        puf = PersistentUnionFind(n)
        edge_list = [[0, 1, 2], [1, 2, 4], [2, 0, 8], [1, 0, 16]]
        edge_list.sort(key=lambda item: item[2])
        for x, y, tm in edge_list:
            puf.union(x, y, tm)
        queries = [[0, 1, 2], [0, 2, 5]]
        assert [puf.is_connected(x, y, tm) for x, y, tm in queries] == [False, True]


if __name__ == '__main__':
    unittest.main()
