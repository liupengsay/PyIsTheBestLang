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
    def maxNumOfSubstrings(self, s: str) -> List[str]:

        n = len(s)
        dct = defaultdict(list)
        for i in range(n):
            if len(dct[s[i]]) == 2:
                dct[s[i]][1] = i
            else:
                dct[s[i]] = [i, i]

        lst = [dct[w] for w in dct]
        res = set()
        for a, b in lst:

            while True:
                x, y = a, b
                for i in range(a, b+1):
                    c, d = dct[s[i]]
                    x = min(x, c)
                    y = max(y, d)
                if (x, y) == (a, b):
                    break
                a, b = x, y
            res.add((a, b))
        res = [list(r) for r in res]
        res.sort()
        length = [b-a+1 for a, b in res]

        m = len(res)
        dp = [0]*m
        cnt = [0]*m
        ans = [[] for _ in range(m)]
        for i in range(m):
            dp[i] = 1
            ans[i] = [res[i]]
            cnt[i] = length[i]
            for j in range(i):
                if res[j][1] < res[i][0] and (dp[j] + 1 > dp[i] or (dp[j]+1==dp[i] and cnt[j]+length[i]<cnt[i])):
                    dp[i] = dp[j] + 1
                    ans[i] = ans[j] + [res[i]]
                    cnt[i] = cnt[j]+length[i]
        x = dp.index(max(dp))
        ret = []
        for a, b in ans[x]:
            ret.append(s[a:b+1])
        return ret





assert Solution().maxNumOfSubstrings("adefaddaccc") == ["e","f","ccc"]
assert Solution().maxNumOfSubstrings(s = "abbaccd") == ["bb","cc", "d"]
