import bisect
import random
import re
import sys
import unittest
from typing import List, Callable
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

from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList

sys.set_int_max_str_digits(0)


class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        if x == 0:
            return 0
        lst = SortedList()
        j = 0
        n = len(nums)
        ans = inf
        for i in range(n):
            while j < n and j <= i - x:
                lst.add(nums[j])
                j += 1
            ind = lst.bisect_left(nums[i])
            for k in [ind - 1, ind, ind + 1]:
                if 0 <= k < len(lst) and abs(nums[i] - lst[k]) < ans:
                    ans = abs(nums[i] - lst[k])
        return ans


assert Solution()
