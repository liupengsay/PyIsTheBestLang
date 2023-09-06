import bisect
import random
import re
import sys
import unittest
from typing import List, Callable
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache, cmp_to_key
from itertools import combinations, accumulate, chain
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop, heapify
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList

# sys.set_int_max_str_digits(0)  # 大数的范围坑


def ac_max(a, b):
    return a if a > b else b


def ac_min(a, b):
    return a if a < b else b




class Solution:
    def maximumBobPoints(self, numArrows: int, aliceArrows: List[int]) -> List[int]:
        n = len(aliceArrows)
        ans = [0]*n
        ans[0] = numArrows
        res = 0
        for i in range(1<<n):
            lst = [0]*n
            cur = 0
            for j in range(n):
                if i & (1<<j):
                    lst[j] = aliceArrows[j] + 1
                    cur += j
            s = sum(lst)
            if s <= numArrows:
                lst[0] += numArrows - s
                if cur > res:
                    res = cur
                    ans = lst[:]
        return ans





assert Solution()
