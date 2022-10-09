

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict

from functools import lru_cache

import random
from itertools import permutations
import numpy as np

from decimal import Decimal

import heapq
import copy


class Solution:
    def maxHappyGroups(self, batchSize: int, groups: List[int]) -> int:

        @lru_cache(None)
        def dfs(tup):
            # print(tup)
            lst = list(tup)

            res = 0
            for x in range(1, batchSize):
                y = batchSize - x
                if x != y:
                    add = lst[x] if lst[x] < lst[y] else lst[y]
                    res += add
                    lst[x] -= add
                    lst[y] -= add

            if sum(lst) == 0:
                return res

            def check(i):
                nonlocal ans
                if i == batchSize:
                    if sum(pre) > 0 and sum(pre[i]*i for i in range(batchSize)) % batchSize == 0:
                        ans = max(ans, 1 + dfs(tuple([lst[j] - pre[j] for j in range(batchSize)])))
                    return
                for num in range(lst[i] + 1):
                    pre[i] = num
                    check(i + 1)
                    pre[i] = 0
                return
            ans = 1
            pre = [0] * batchSize
            check(0)
            return ans + res

        cnt = [0] * batchSize
        for g in groups:
            cnt[g % batchSize] += 1

        final = cnt[0]
        cnt[0] = 0
        final += dfs(tuple(cnt))
        #print(final)
        return final


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().maxHappyGroups(batchSize = 3, groups = [1,2,3,4,5,6]) == 4
        assert Solution().maxHappyGroups(batchSize = 4, groups = [1,3,2,5,2,2,1,6]) == 4
        assert Solution().maxHappyGroups(3, [844438225,657615828,355556135,491931377,644089602,30037905,863899906,246536524,682224520]) == 6
        assert Solution().maxHappyGroups(8, [244197059,419273145,329407130,44079526,351372795,200588773,340091770,851189293,909604028,621703634,959388577,989293607,325139045,263977422,358987768,108391681,584357588,656476891,621680874,867119215,639909909,98831415,263171984,236390093,21876446]) == 13
        return


if __name__ == '__main__':
    unittest.main()
