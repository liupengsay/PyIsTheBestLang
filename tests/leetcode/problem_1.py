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





@lru_cache(None)
def dfs(n, k):
    if not n:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 1
    if n == k:
        return 1
    if not dfs(n-1, k) or not dfs(n-2, k):
        return 1
    if n > k and not dfs(n-k, k):
        return 1
    return 0


x = 96
for k in range(3, x):
    print(x, k, x%3, k%3, dfs(x, k))



#
# assert Solution()