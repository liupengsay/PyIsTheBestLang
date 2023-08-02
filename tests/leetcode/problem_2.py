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
    def minFlips(self, s: str) -> int:
        n = len(s)
        s += s
        s1 = "10" * n
        s2 = "01" * n
        pre1 = list(accumulate([int(s1[i]!=s[i]) for i in range(2*n)], initial=0))
        pre2 = list(accumulate([int(s2[i]!=s[i]) for i in range(2*n)], initial=0))
        ans1 = min(pre1[i+n]-pre1[i] for i in range(n))
        ans2 = min(pre2[i + n] - pre2[i] for i in range(n))
        return ans1 if ans1 < ans2 else ans2


assert Solution().minFlips("010") == 0



