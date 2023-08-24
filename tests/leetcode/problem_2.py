import bisect
import random
import re
import sys
import unittest
from typing import List, Callable
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache, cmp_to_key
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

import heapq
import copy
from sortedcontainers import SortedList

# sys.set_int_max_str_digits(0)  # 大数的范围坑



class FrequencyTracker:

    def __init__(self):
        self.cnt = defaultdict(int)
        self.freq = defaultdict(int)

    def add(self, number: int) -> None:
        pre = self.cnt[number]
        self.cnt[number] += 1
        self.freq[pre+1] += 1
        if pre:
            self.freq[pre] -= 1
        return

    def deleteOne(self, number: int) -> None:
        pre = self.cnt[number]
        if not pre:
            return
        self.cnt[number] -= 1
        self.freq[pre] -= 1
        if pre - 1:
            self.freq[pre - 1] += 1
        return


    def hasFrequency(self, frequency: int) -> bool:
        return self.freq[frequency] > 0







assert Solution()