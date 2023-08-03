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
    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:

        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        ans = [0]*n

        def dfs(x, fa):
            cnt = [0]*26
            for y in dct[x]:
                if y != fa:
                    nex = dfs(y, x)
                    for i in range(26):
                        cnt[i] += nex[i]

            cnt[ord(labels[x])-ord("a")] += 1
            ans[x] = cnt[ord(labels[x])-ord("a")]
            return cnt

        dfs(0, -1)
        return ans


assert Solution().minFlips("010") == 0



