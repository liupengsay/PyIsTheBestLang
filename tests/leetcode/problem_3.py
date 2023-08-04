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





mod = 10**9 + 7

dct = [[] for _ in range(6)]


def dfs():
    global pre
    if pre:
        dct[len(pre)].append(pre[:])
    if len(pre) == 5:
        return
    for x in range(3):
        if not pre or pre[-1] != x:
            pre.append(x)
            dfs()
            pre.pop()
    return

pre = []
dfs()


edge = [[]]
for x in range(1, 6):
    lst = dct[x][:]
    m = len(lst)
    cur = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i+1, m):
            if all(lst[i][y] != lst[j][y] for y in range(x)):
                cur[i].append(j)
                cur[j].append(i)
    edge.append(copy.deepcopy(cur))


@lru_cache(None)
def dfs(m, n, s):
    if n == 1:
        return 1

    res = 0
    for j in cur[m][s]:
        res += dfs(m, n-1, j)
        res %= mod
    return res


class Solution:
    def colorTheGrid(self, m: int, n: int) -> int:
        return sum(dfs(m, n, x) for x in range(len(cur[m]))) % mod






assert Solution().minOperations(nums1 = [6,6], nums2 = [1]) == 3






