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
    def countPalindromicSubsequence(self, s: str) -> int:

        n = len(s)
        pre = [set() for _ in range(n)]
        cur = set()
        for i in range(n):
            pre[i] = cur
            cur.add(s[i])

        ans = set()
        cur = set()
        for i in range(n-1, -1, -1):
            for w in cur:
                if w in pre[i]:
                    ans.add(w+s[i]+w)
            cur.add(s[i])
        return len(ans)





assert Solution().closestCost(baseCosts = [2,3], toppingCosts = [4,5,100], target = 18) == 17



