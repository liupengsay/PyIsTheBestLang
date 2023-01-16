

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from operator import xor, mul, add
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy



# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = defaultdict(list)

        def dfs(node, i, j):
            nonlocal ind
            if not node:
                return
            ans[j].append([i, ind, node.val])
            ind += 1
            dfs(node.left, i+1, j - 1)
            dfs(node.right, i+1, j + 1)
            return

        ind = 0
        dfs(root, 0, 0)
        axis= list(ans.keys())
        for i in axis:
            ans[i].sort(key=lambda x: [x[0], x[1]])
        return [[val for _, _, val in ans[i]] for i in sorted(ans)]


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countPairs(nums=[1, 4, 2, 7], low=2, high=6) == 6
        return


if __name__ == '__main__':
    unittest.main()
