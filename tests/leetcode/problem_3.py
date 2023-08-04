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
    def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
        dct = [0 for _ in range(n)]
        for i, j in roads:
            dct[i] += 1
            dct[j] += 1

        ind = list(range(n))
        ind.sort(key=lambda it: dct[it])
        value = [0]*n
        for i in range(n):
            value[ind[i]] = i+1

        ans = 0
        for i, j in roads:
            ans += value[i]+value[j]
        return ans






assert Solution().minOperations(nums1 = [6,6], nums2 = [1]) == 3






