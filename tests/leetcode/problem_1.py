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
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        dct = {w: chr(i+ord("a")) for i, w in enumerate(order)}
        words = ["".join(dct[w] for w in word) for word in words]
        return words == sorted(words)






assert Solution()