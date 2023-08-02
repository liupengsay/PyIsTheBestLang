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

mod = 10 ** 9 + 7


class Solution:
    def minWastedSpace(self, packages: List[int], boxes: List[List[int]]) -> int:
        ans = inf
        packages.sort()
        pre = list(accumulate(packages, initial=0))
        n = len(packages)
        for box in boxes:
            box.sort()

            if box[-1] < packages[-1]:
                continue
            cur = i = 0
            for num in box:
                j = bisect.bisect_left(packages, num)
                cur += num*(j-i) - (pre[j+1]-pre[i+1])
                i = j
            if cur < ans:
                ans = cur
        return ans % mod if ans < inf else -1




assert Solution().kSimilarity("ab", "ba") == 1
