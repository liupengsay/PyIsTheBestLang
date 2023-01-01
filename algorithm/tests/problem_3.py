


import bisect
import itertools
import random
from typing import List
import heapq
import math
import re
import unittest
from collections import defaultdict, Counter, deque
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq
from itertools import combinations, permutations



class Solution:
    def minimumPartition(self, s: str, k: int) -> int:

        if any(int(w) > k for w in s):
            return -1

        ans = 0
        pre = ""
        for w in s:
            if int(pre + w) <= k:
                pre += w
            else:
                ans += 1
        return ans + 1





class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().isPossible(4, [[1,2],[2,3],[2,4],[3,4]]) == False

        return


if __name__ == '__main__':
    unittest.main()
