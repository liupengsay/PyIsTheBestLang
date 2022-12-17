

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


class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:

        ans = -1
        def dfs(node):
            nonlocal ans
            if not node:
                return [0, 0]
            left = dfs(node.left)
            right = dfs(node.right)
            ans = max(ans, left[1]+1, right[0]+1)
            res = [left[1]+1, right[0]+1]
            return res
        dfs(root)
        return ans

class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().minCost(7, [1, 3, 4, 5]) == 11
        return


if __name__ == '__main__':
    unittest.main()
