
import bisect
import random
import re
import unittest
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, permutations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq

import random
from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from functools import cmp_to_key

from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from functools import cmp_to_key

from sortedcontainers import SortedList
from operator import mul
from functools import reduce

from functools import cmp_to_key


class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        x = nums.index(k)
        pre = defaultdict(int)
        pre[0] = 1
        a = b = 0
        for i in range(x-1, -1, -1):
            if nums[i] > k:
                a += 1
            else:
                b += 1
            pre[a-b] += 1

        post = defaultdict(int)
        post[0] = 1
        a = b = 0
        for i in range(x+1, n):
            if nums[i] > k:
                a += 1
            else:
                b += 1
            post[a - b] += 1

        ans = 0
        for d in pre:
            ans += pre[d]*post[-d]
            ans += pre[d]*(post[-(d-1)])
        return ans


class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countSubarrays(nums = [2,3,1], k = 3) == 1
        assert Solution().countSubarrays(nums = [3,2,1,4,5], k = 4) == 3
        assert Solution().countSubarrays([2,5,1,4,3,6], 1) == 3
        return


if __name__ == '__main__':
    unittest.main()
