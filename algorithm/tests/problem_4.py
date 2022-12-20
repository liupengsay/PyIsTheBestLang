
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
    def cycleLengthQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        ans = []

        for a, b in queries:

            ans_b = [b]
            while b > 1:
                ans_b.append(b // 2)
                b //= 2
            if ans_b[-1] != b:
                ans_b.append(b)
            ind_b = {num: i for i, num in enumerate(ans_b)}

            ans_a = [a]
            while ans_a[-1] not in ind_b:
                ans_a.append(a // 2)
                a //= 2
            ans.append(len(ans_a) + ind_b[ans_a[-1]])
        return ans

class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countSubarrays(nums=[2, 3, 1], k=3) == 1
        assert Solution().countSubarrays(nums=[3, 2, 1, 4, 5], k=4) == 3
        assert Solution().countSubarrays([2, 5, 1, 4, 3, 6], 1) == 3
        return


if __name__ == '__main__':
    unittest.main()
