

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
    def checkDistances(self, s: str, distance: List[int]) -> bool:
        n = len(s)
        pre = dict()
        for i in range(n):
            if s[i] in pre:
                if distance[ord(s[i]) - ord('a')] != i - pre[s[i]] - 1:
                    return False
            else:
                pre[s[i]] = i
        return True


def test_solution():
    assert Solution().checkDistances(
        "abaccb", [
            1, 3, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return


if __name__ == '__main__':
    test_solution()
