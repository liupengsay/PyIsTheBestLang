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
    def minDeletionSize(self, strs: List[str]) -> int:
        ans = 0
        while sorted(strs) != strs:
            n = len(strs[0])
            for j in range(n):
                lst = [word[:j+1] for word in strs]
                if lst != sorted(lst):
                    strs = [word[:j]+word[j+1:] for word in strs]
                    ans += 1
                    break
        return ans





assert Solution().countSubstrings(s = "aba", t = "baba") == 6
