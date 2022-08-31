
import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict

MOD = 10**9 + 7


class Solution:
    def createSortedArray(self, instructions: List[int]) -> int:
        ans = 0
        lst = SortedList()
        for num in instructions:
            n = len(lst)
            i = lst.bisect_left(num)
            j = lst.bisect_right(num)
            lst.add(num)
            ans += i if i < n - j else n - j
            ans %= MOD
        return ans


print(Solution().kthSmallestPath([2, 3], 1) == "HHHVV")
print(Solution().kthSmallestPath([2, 3], 2) == "HHVHV")
print(Solution().kthSmallestPath([2, 3], 3) == "HHVVH")
