
import bisect
import itertools
import random
from typing import List
import heapq
import math
import re
import unittest
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq
from itertools import combinations
from sortedcontainers import SortedList


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


class Solution:
    def minimumOperations(self, root: Optional[TreeNode]) -> int:

        def check(lst):
            val = [node.val for node in lst]
            tmp = sorted(val)
            ind = {num: i for i, num in enumerate(val)}
            n = len(val)
            uf = UnionFind(n)
            for i in range(n):
                uf.union(ind[val[i]], ind[tmp[i]])
            return sum(p - 1 for p in uf.size if p >= 1)

        ans = 0
        stack = [root]
        while stack:
            ans += check(stack)
            nex = []
            for node in stack:
                if node.left:
                    nex.append(node.left)
                if node.right:
                    nex.append(node.right)
            stack = nex[:]
        return ans


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution()
        return


if __name__ == '__main__':
    unittest.main()
