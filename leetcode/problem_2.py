
import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet
from itertools import combinations
from sortedcontainers import SortedDict

import heapq
from sortedcontainers import SortedList

class Solution:
    def canEat(self, candiesCount: List[int], queries: List[List[int]]) -> List[bool]:
        n = len(candiesCount)
        pre =  [0]
        for i in range(n):
            pre.append(pre[-1]+candiesCount[i])

        ans = []


def test_solution():
    assert Solution().minCharacters("dee", "a") == 0
    return


if __name__ == '__main__':
    test_solution()
