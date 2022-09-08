
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


MOD = 10**9 + 7


class Solution:
    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:

        pre = defaultdict()
        pre[startPos] = 1
        while k:
            nex = defaultdict()
            for pos in pre:
                nex[pos + 1] += pre[pos]
                nex[pos - 1] += pre[pos]
            for pos in nex:
                nex[pos] %= MOD
            pre = nex.copy()
            k -= 1
        return pre[endPos] % MOD


def test_solution():
    assert Solution().countPairs([1, 3, 5, 7, 9]) == 4
    return


if __name__ == '__main__':
    test_solution()
