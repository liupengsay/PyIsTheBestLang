

import bisect
import random

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict

from functools import lru_cache

import random
from itertools import permutations
import numpy as np

from decimal import Decimal


import heapq
import copy
from collections import defaultdict

class Solution:
    def countEven(self, num: int) -> int:
        ans = 0
        for i in range(1, num+1):
            if sum([int(w) for w in str(i)]) % 2==0:
                ans += 1
        return ans


def test_solution():
    assert Solution().maximumANDSum([1,2,3,4,5,6], 3) == 9
    return


if __name__ == '__main__':
    test_solution()
