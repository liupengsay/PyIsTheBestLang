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
    def canReorderDoubled(self, arr: List[int]) -> bool:
        pos = [num for num in arr if num > 0]
        neg = [num for num in arr if num < 0]
        zero = arr.count(0)
        if zero % 2:
            return False

        cnt = Counter(pos)
        pos.sort()
        for num in pos:
            if not cnt[num]:
                continue
            x = cnt[num]
            if cnt[2*num] < x:
                return False
            cnt[2*num] -= x

        cnt = Counter(neg)
        neg.sort(reverse=True)
        for num in neg:
            if not cnt[num]:
                continue
            x = cnt[num]
            if cnt[2 * num] < x:
                return False
            cnt[2 * num] -= x
        return True




assert Solution()



