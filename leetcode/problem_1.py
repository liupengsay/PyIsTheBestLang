

import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict


class Solution:
    def countBalls(self, lowLimit: int, highLimit: int) -> int:
        cnt = defaultdict(int)
        for num in range(lowLimit, highLimit+1):
            k = sum([int(s) for s in str(num)])
            cnt[k] += 1
        c = max(cnt.values())
        return [k for k in cnt if cnt[k]==c][0]



def test_solution():
    assert Solution().checkDistances(
        "abaccb", [
            1, 3, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return


if __name__ == '__main__':
    test_solution()
