import bisect
import random
import re
import unittest
from bisect import bisect_left

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np
from typing import List, Callable
from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList


def dijkstra_src_to_dst_path(dct, src: int, dst: int):
    # 模板: Dijkstra求起终点的最短路，注意只能是正权值可以提前返回结果，并返回对应经过的路径
    n = len(dct)
    dis = [inf] * n
    stack = [[0, src]]
    dis[src] = 0
    father = [-1] * n  # 记录最短路的上一跳
    while stack:
        d, i = heapq.heappop(stack)
        if dis[i] < d:
            continue
        if i == dst:
            break
        for j in dct[i]:
            dj = dct[i][j] + d
            if dj < dis[j]:
                dis[j] = dj
                father[j] = i
                heapq.heappush(stack, [dj, j])
    # 向上回溯路径
    path = []
    i = dst
    while i != -1:
        path.append(i)
        i = father[i]
    return path, dis[dst]



# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:

        dct = defaultdict(list)

        def dfs(node):
            if not node:
                return
            x = node.val
            if node.left:
                y = node.left.val
                dct[x].append([y, "L"])
                dct[y].append([x, "U"])
                dfs(node.left)

            if node.right:
                y = node.right.val
                dct[x].append([y, "R"])
                dct[y].append([x, "U"])
                dfs(node.right)

            return

        dfs(root)

        parent = defaultdict(list)
        stack = [[startValue, -1]]
        while stack:
            i, fa = stack.pop()
            for j, s in dct[i]:
                if j != fa:
                    parent[j] = [i, s]
                    stack.append([j, i])

        x = destValue
        ans = ""
        while x != startValue:
            x, s = parent[x]
            ans += s
        return ans









assert Solution()
