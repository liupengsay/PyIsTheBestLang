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
    def indexPairs(self, text: str, words: List[str]) -> List[List[int]]:

        dct = set(words)
        ans = []
        n = len(text)
        for i in range(n):
            for j in range(i, n):
                if text[i:j+1] in dct:
                    ans.append([i, j])
        return ans








assert Solution()



