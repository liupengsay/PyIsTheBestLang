
import bisect

from typing import List

import math
from collections import defaultdict, Counter
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

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
    def connectTwoGroups(self, cost: List[List[int]]) -> int:
        m, n = len(cost), len(cost[0])
        res = [[10**9]*(m+n) for _ in range(m+n)]
        edges = []
        dct = defaultdict(dict)
        for i in range(m):
            for j in range(n):
                edges.append([i, m+j, cost[i][j]])
                res[i][m+j] = cost[i][j]
                res[m+j][i] = cost[i][j]
        print(res)
        return 0


#assert Solution().connectTwoGroups([[15, 96], [36, 2]]) == 17
assert Solution().connectTwoGroups([[1, 3, 5], [4, 1, 1], [1, 5, 3]]) == 4
assert Solution().connectTwoGroups(
    [[2, 5, 1], [3, 4, 7], [8, 1, 2], [6, 2, 4], [3, 8, 8]]) == 10
