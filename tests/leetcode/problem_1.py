import bisect
import random
import re
import unittest

from typing import List, Callable
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
from typing import List, Callable
import heapq
import copy
from sortedcontainers import SortedList



class Solution:
    def findEvenNumbers(self, digits: List[int]) -> List[int]:

        ans = set()
        for item in permutations(digits, 3):
            a, b, c = item
            if a == 0 or c % 2:
                continue
            ans.add(a*100+b*10+c)
        ans = list(ans)
        ans.sort()
        return ans





assert Solution()