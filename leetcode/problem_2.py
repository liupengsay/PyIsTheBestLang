
import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet
from itertools import combinations
from sortedcontainers import SortedDict

from sortedcontainers import SortedList, SortedDict, SortedSet
from itertools import combinations, permutations
from sortedcontainers import SortedDict
from decimal import Decimal

from collections import deque




class Solution:
    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        n = len(s)
        if n < k or k > 26:
            return 0

        ans = 0
        dct = defaultdict(int)
        for i in range(k-1):
            dct[s[i]] += 1
        for i in range(k-1, n):
            dct[s[i]] += 1
            if i-k >= 0:
                dct[s[i-k]] -= 1
            if max(dct.values()) <= 1:
                ans += 1
        return ans




if __name__ == '__main__':
    test_solution()
