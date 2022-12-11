
import bisect
import random
import re
import unittest
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, permutations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq

import random
from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from functools import cmp_to_key

from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from functools import cmp_to_key

from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from functools import cmp_to_key


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
    def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:

        # 根据邻居关系进行建图处理
        dct = []
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
        ans = []
        j = 0
        length = len(dct)
        for i in ind:
            cur = queries[i]
            while j < length and dct[j][2] < cur:
                uf.union(dct[j][0], dct[j][1])
                j += 1
            if cur > grid[0][0]:
                ans.append([i, uf.size[uf.find(0)]])
            else:
                ans.append([i, 0])
        ans.sort(key=lambda a: a[0])
        return [a[1] for a in ans]


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countSubarrays(nums=[2, 3, 1], k=3) == 1
        assert Solution().countSubarrays(nums=[3, 2, 1, 4, 5], k=4) == 3
        assert Solution().countSubarrays([2, 5, 1, 4, 3, 6], 1) == 3
        return


if __name__ == '__main__':
    unittest.main()
