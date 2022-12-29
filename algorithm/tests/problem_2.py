
import bisect
import re
import unittest
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

from sortedcontainers import SortedList



class Solution:
    def takeCharacters(self, s: str, k: int) -> int:
        cnt= Counter(s)
        if cnt["a"]<k or cnt["b"]<k or cnt["c"]<k:
            return -1

        n = len(s)

        if any(cnt[w]==k for w in "abc"):
            return n

        def check(length):
            cur = defaultdict(int)
            for i in range(length-1):
                cur[s[i]] += 1
            for i in range(length-1, n):
                cur[s[i]] += 1
                if all(cnt[w]-cur[w]>=k for w in "abc"):
                    return True
                cur[s[i-length+1]] -= 1
            return False


        low = 1
        high = n
        while low < high-1:
            mid = low+(high-low)//2
            if check(mid):
                low = mid
            else:
                high = mid
        ans = high if check(high) else low
        return n-ans





class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().smallestValue(4) == 4
        return


if __name__ == '__main__':
    unittest.main()
