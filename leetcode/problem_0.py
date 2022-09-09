import bisect

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import random


"""
# Definition for an Interval.
class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end
"""


from sortedcontainers import SortedList


class Solution:
    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        cnt = defaultdict(int)
        stack = []
        null = SortedList(list(range(k)))
        for i, start in enumerate(arrival):
            while stack and stack[0][0] <= start:
                _, ind = heapq.heappop(stack)
                null.add(ind)
            if null:
                j = i % k
                p = null.bisect_left(j)
                if 0 <= p < len(null):
                    cnt[null[p]] += 1
                    heapq.heappush(stack, [start + load[i], null[p]])
                else:
                    cnt[null[0]] += 1
                    heapq.heappush(stack, [start + load[i], null[0]])
        target = max(cnt.values())
        return [k for k in cnt if cnt[k] == target]


def test_solution():
    assert Solution().minDistance([2, 3, 5, 12, 18], 2) == 9
    return


if __name__ == '__main__':
    test_solution()
