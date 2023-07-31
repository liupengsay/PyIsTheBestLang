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



mod = 10**9 + 7
class Solution:
    def maxSumMinProduct(self, nums: List[int]) -> int:

        n = len(nums)
        lst = list(accumulate(nums, initial=0))

        post = [n-1] * n  # [n-1] * n

        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                post[stack.pop()] = i-1  # i - 1
            stack.append(i)

        pre = [0] * n  # [0] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]:
                pre[stack.pop()] = i + 1  # i - 1
            stack.append(i)

        ans = 0

        for i in range(n):
            x = nums[i]
            left= pre[i]
            right = post[i]
            cur = x*(lst[right+1]-lst[left])
            if cur > ans:
                ans = cur
        return ans % mod


