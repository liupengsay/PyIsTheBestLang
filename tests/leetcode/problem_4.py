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

    def kSimilarity(self, s1: str, s2: str) -> int:

        visit = {s1}
        n = len(s1)
        stack = deque([(s1, 0)])
        while stack:
            s, step = stack.popleft()
            if s == s2:
                return step
            lst = list(s)
            for i in range(n):
                if lst[i] != s2[i]:
                    for j in range(i+1, n):
                        if lst[j] != s2[j] and lst[j] == s2[i]:
                            lst[i], lst[j] = lst[j], lst[i]
                            st = "".join(lst)
                            if st not in visit:
                                visit.add(st)
                                stack.append((st, step+1))
                            lst[i], lst[j] = lst[j], lst[i]
        return -1

        return dfs(s1, s2)




assert Solution().kSimilarity("ab", "ba") == 1
