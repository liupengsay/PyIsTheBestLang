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




mod = 10**9 + 7


class Solution:
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:

        n = len(nums1)
        ans = sum(abs(nums1[i]-nums2[i]) for i in range(n))

        lst = SortedList(nums1)
        for i in range(n):
            x = nums2[i]
            y = nums1[i]
            j = lst.bisect_left(x)
            cur = ans - abs(x-y)
            for k in [j-1, j]:
                if 0<=k<n:
                    if cur + abs(lst[k]-x) < ans:
                        ans = cur + abs(lst[k]-x)


        return ans % mod





assert Solution().minOperations(nums1 = [6,6], nums2 = [1]) == 3






