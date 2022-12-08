
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
        part = defaultdict(set)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].add(i)
        return part

    def get_root_size(self):
        # 获取每个根节点对应的组大小
        size = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
        return size


class Solution:
    def magnificentSets(self, n: int, edges: List[List[int]]) -> int:
        dct = [[] for _ in range(n)]
        degree = defaultdict(int)
        uf = UnionFind(n)
        for i, j in edges:
            i -= 1
            j -= 1
            dct[i].append(j)
            dct[j].append(i)
            degree[i]+=1
            degree[j] += 1
            uf.union(i, j)

        part = uf.get_root_part()

        def check(nodes, edges):
            res = 0

            for i in nodes:
                root = defaultdict(lambda: int)
                ans = 0
                visit = [0] * n
                if not visit[i]:
                    stack = [i]
                    visit[i] = 1
                    while stack:
                        nex = []
                        for x in stack:
                            root[x] = ans
                        ans += 1
                        for x in stack:
                            for y in dct[x]:
                                if not visit[y]:
                                    nex.append(y)
                                    visit[y] = 1
                        stack = nex
                if all(abs(root[x-1]-root[y-1])==1 for x, y in edges):
                    res = max(res, len(set(root)))
            return res
        ans = 0
        for p in part:
            cur = check(list(part[p]), [[x-1,y-1] for x, y in edges if x-1 in part[p] and y-1 in part[p]])
            ans = max(ans, cur)
        return ans if ans> 0 else -1




class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countSubarrays(nums = [2,3,1], k = 3) == 1
        assert Solution().countSubarrays(nums = [3,2,1,4,5], k = 4) == 3
        assert Solution().countSubarrays([2,5,1,4,3,6], 1) == 3
        return


if __name__ == '__main__':
    unittest.main()
