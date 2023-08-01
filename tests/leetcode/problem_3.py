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

from sortedcontainers import SortedList





class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        n = len(workers)
        m = len(bikes)

        @lru_cache(None)
        def dfs(i, state):
            if i == n:
                return 0
            x, y = workers[i]
            res = inf
            for j in range(m):
                if state & (1<<j):
                    a, b = bikes[j]
                    cur = abs(x-a)+abs(b-y) + dfs(i+1, state^(1<<j))
                    if cur < res:
                        res = cur
            return res

        return dfs(0, (1<<m)-1)



assert Solution().countSubstrings(s = "aba", t = "baba") == 6
