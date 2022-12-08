
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
    def dividePlayers(self, skill: List[int]) -> int:
        s = sum(skill)
        n = len(skill)

        if s % (n//2) != 0:
            return -1

        m = s//(n//2)
        pre = defaultdict(int)
        ans =0
        for num in skill:
            if pre[m-num]:
                ans += num*(m-num)
                pre[m-num] -= 1
            else:
                pre[num] +=1
        if sum(pre.values()) == 0:
            return ans
        return -1






class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().minSwap(nums1=[1, 3, 5, 4], nums2=[1, 2, 3, 7]) == 1
        return


if __name__ == '__main__':
    unittest.main()
