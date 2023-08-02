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
    def minFlips(self, s: str) -> int:
        n = len(s)
        ans = n

        for w in [1, 0]:
            pre = w
            t = str(pre)
            for _ in s[1:]:
                t += str(1-pre)
                pre = 1-pre
            cur = sum(int(s[i]!=t[i]) for i in range(n))
            if cur < ans:
                ans = cur
        s = s[1:] + s[0]
        for w in [1, 0]:
            pre = w
            t = str(pre)
            for _ in s[1:]:
                t += str(1-pre)
                pre = 1-pre
            cur = sum(int(s[i]!=t[i]) for i in range(n))
            if cur < ans:
                ans = cur
        return ans







assert Solution().countSubstrings(s = "aba", t = "baba") == 6
