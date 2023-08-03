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






class Solution:
    def maxSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        ans= 0
        ind = [[-1, -1], [-1, 0], [-1, 1], [0, 0], [1, -1], [1, 0], [1, 1]]
        for i in range(1, m-1):
            for j in range(1, n-1):
                cur = sum(grid[i+a][j+b] for a, b in ind)
                if cur > ans:
                    ans = cur
        return ans




assert Solution().snakesAndLadders([[-1,-1,-1,46,47,-1,-1,-1],[51,-1,-1,63,-1,31,21,-1],[-1,-1,26,-1,-1,38,-1,-1],[-1,-1,11,-1,14,23,56,57],[11,-1,-1,-1,49,36,-1,48],[-1,-1,-1,33,56,-1,57,21],[-1,-1,-1,-1,-1,-1,2,-1],[-1,-1,-1,8,3,-1,6,56]]) == 4



