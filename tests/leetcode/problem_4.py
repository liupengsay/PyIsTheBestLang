import bisect
import random
import re
import unittest

from typing import List, Callable, Dict, Optional
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




class NumberTheoryAllFactor:
    def __init__(self, ceil):
        self.ceil = ceil+10
        self.factor = [[1] for _ in range(self.ceil+1)]
        self.get_all_factor()
        return

    def get_all_factor(self):
        # 模板：计算 1 到 self.ceil 所有数字的所有因子
        for i in range(2, self.ceil + 1):
            x = 1
            while x*i <= self.ceil:
                self.factor[x*i].append(i)
                x += 1
        return


nt = NumberTheoryAllFactor(2*10**5+10)
class Solution:
    def countDifferentSubsequenceGCDs(self, nums: List[int]) -> int:

        dct = defaultdict(list)
        for num in set(nums):
            for x in nt.factor[num]:
                dct[num].append(x)
        ans = 0
        for num in dct:
            if reduce(math.gcd, dct[num]) == num:
                ans += 1
        return ans


nums = list(range(1, 2*10**5+1))

assert Solution().countDifferentSubsequenceGCDs(nums)
