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
    def findingUsersActiveMinutes(self, logs: List[List[int]], k: int) -> List[int]:

        dct = defaultdict(set)
        for id, item in logs:
            dct[id].add(item)
        cnt = Counter([len(dct[k]) for k in dct])
        return [cnt[x] for x in range(1, k+1)]





assert Solution().closestCost(baseCosts = [2,3], toppingCosts = [4,5,100], target = 18) == 17



