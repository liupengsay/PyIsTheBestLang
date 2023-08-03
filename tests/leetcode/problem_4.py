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
    def closestToTarget(self, arr: List[int], target: int) -> int:

        ans = set()
        pre = set()
        for num in arr:
            pre = pre | {p&num for p in pre}
            pre.add(num)
            ans |= pre
        return min(abs(x-target) for x in ans)





assert Solution().closestToTarget(arr = [1,2,4,8,16], target = 0) == 0
