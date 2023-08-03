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
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        ans = numBottles
        cur = numBottles
        while cur >= numExchange:
            ans += cur//numExchange
            cur = cur % numExchange + cur // numExchange
        return ans



assert Solution().maxJumps(nums = [6,4,14,6,8,13,9,7,10,6,12], d = 2) == 4