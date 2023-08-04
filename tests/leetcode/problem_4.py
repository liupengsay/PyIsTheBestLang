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
    def earliestAndLatest(self, n: int, firstPlayer: int, secondPlayer: int) -> List[int]:


        state = (1<<n)-1
        a = firstPlayer-1
        b = secondPlayer-1

        @lru_cache(None)
        def dfs(s):
            if not s &(1<<a) or not s&(1<<b):
                return [inf, -inf]
            lst = [i for i in range(n) if s&(1<<i)]
            m = len(lst)
            for j in range(m//2):
                if lst[j] == firstPlayer and lst[m-1-j] == secondPlayer:
                    return [0, 0]

            res = [inf, -inf]
            for x in range(1<<(m//2)):
                nex_s = 0
                for j in range(m//2):
                    if x & (1<<j):
                        nex_s |= (1<<lst[j])
                    else:
                        nex_s |= (1 << lst[m-1-j])
                if m % 2:
                    nex_s |= (1<<lst[m//2])
                cur = dfs(nex_s)
                if cur[0]+1<res[0]:
                    res[0] = cur[0]+1
                if cur[1]+1>res[1]:
                    res[1] = cur[1]+1
            return res

        return dfs(state)




assert Solution().deleteString(st) == 4000
