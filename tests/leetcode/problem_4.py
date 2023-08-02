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
    def tallestBillboard(self, rods: List[int]) -> int:
        n = len(rods)

        pre = defaultdict(int)
        pre[0] = 0
        for num in rods:
            cur = defaultdict(int)
            for p in pre:
                cur[p+num] = max(cur[p+num], pre[p])
                cur[p - num] = max(cur[p - num], pre[p]+num)
            pre = cur.copy()
        return pre[0]




assert Solution().kSimilarity("ab", "ba") == 1
