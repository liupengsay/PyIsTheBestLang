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





class Solution:
    def minimizeXor(self, num1: int, num2: int) -> int:
        m = bin(num2).count("1")

        lst = [0]*60
        ind = [i for i in range(60) if num1&(1<<i)]
        ind.reverse()
        for i in ind:
            if m:
                m -= 1
                lst[i] = 1
            else:
                break
        for i in range(60):
            if m and not lst[i]:
                lst[i] = 1
                m -= 1
        lst.reverse()
        return int("0b"+"".join(str(x) for x in lst), 2)







assert Solution().maxNumOfSubstrings("adefaddaccc") == ["e","f","ccc"]

