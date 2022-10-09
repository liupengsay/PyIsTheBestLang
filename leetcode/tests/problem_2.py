
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
    def findArray(self, pref: List[int]) -> List[int]:
        n = len(pref)
        ans = [pref[0]]
        for i in range(1, n):
            ans.append(pref[i - 1] ^ pref[i])
        return ans

class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().maxHappyGroups(batchSize = 3, groups = [1,2,3,4,5,6]) == 4
        assert Solution().maxHappyGroups(batchSize = 4, groups = [1,3,2,5,2,2,1,6]) == 4
        assert Solution().maxHappyGroups(3, [844438225,657615828,355556135,491931377,644089602,30037905,863899906,246536524,682224520]) == 6
        assert Solution().maxHappyGroups(8, [244197059,419273145,329407130,44079526,351372795,200588773,340091770,851189293,909604028,621703634,959388577,989293607,325139045,263977422,358987768,108391681,584357588,656476891,621680874,867119215,639909909,98831415,263171984,236390093,21876446]) == 13
        return


if __name__ == '__main__':
    unittest.main()
