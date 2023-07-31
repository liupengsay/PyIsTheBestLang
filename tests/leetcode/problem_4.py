import bisect
import random
import re
import unittest

from typing import List, Callable, Dict
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
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:

        n = len(colors)
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            if i == j:
                return -1
            dct[i].append(j)
            degree[j] += 1


        ans = 0
        for color in range(26):
            cnt = [0 for _ in range(n)]
            stack = deque([i for i in range(n) if not degree[i]])
            for i in stack:
                cnt[i] += ord(colors[i])-ord("a") == color
            while stack:
                i = stack.popleft()
                for j in dct[i]:
                    degree[j] -= 1
                    a, b = cnt[j], cnt[i]
                    cnt[j] = a if a > b else b
                    if not degree[j]:
                        cnt[j] += ord(colors[j])-ord("a") == color
                        stack.append(j)
            if not all(x == 0 for x in degree):
                return -1
            cur = max(cnt)
            ans = ans if ans > cur else cur
        return ans




assert Solution().largestNumber(cost = [4,3,2,5,6,7,2,5,5], target = 9) == "7772"
