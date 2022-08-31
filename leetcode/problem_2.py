
import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict


class Solution:
    def minDeletions(self, s: str) -> int:
        cnt = sorted(list(Counter(s).values()), reverse=True)
        ans = 0
        pre = set()
        for num in cnt:
            minus = 0
            while num > minus and num - minus in pre:
                minus += 1
            pre.add(num - minus)
            ans += minus
        return ans


assert Solution().findLexSmallestString("5525", 9, 2) == "2050"
assert Solution().findLexSmallestString("74", 5, 1) == "24"
assert Solution().findLexSmallestString("0011", 4, 2) == "0011"
assert Solution().findLexSmallestString("43987654", 7, 3) == "00553311"
