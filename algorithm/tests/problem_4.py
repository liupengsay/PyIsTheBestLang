
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

mod = 10 ** 9 + 7


class Solution:
    def countPartitions(self, nums: List[int], k: int) -> int:
        s = sum(nums)
        big = [num for num in nums if num >= k]
        small = [num for num in nums if num < k]
        s = sum(small)
        pre = defaultdict(int)
        pre[0] = 1
        for num in small:
            cur = pre.copy()
            for p in pre:
                cur[p + num] += pre[p]
                cur[p + num] %= mod
            pre = cur.copy()

        m = len(big)
        ans = 0
        for w in pre:
            a, b = w, s- w
            if a >= k and b>=k:
                ans += pow(2, m, mod)*pre[w]
            if a < k and b>=k:
                if m:
                    ans += m*pow(2, m-1, mod)*pre[w]
            if a>=k and b<k:
                if m:
                    ans += m * pow(2, m - 1, mod) * pre[w]
            if a<k and b<k:
                if m>=2:
                    ans += m*(m-1)*pow(2, m - 2, mod)*pre[w]
            ans %= mod
        return ans



class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().countPartitions([977208288,291246471,396289084,732660386,353072667,34663752,815193508,717830630,566248717,260280127,824313248,701810861,923747990,478854232,781012117,525524820,816579805,861362222,854099903,300587204,746393859,34127045,823962434,587009583,562784266,115917238,763768139,393348369,3433689,586722616,736284943,596503829,205828197,500187252,86545000,490597209,497434538,398468724,267376069,514045919,172592777,469713137,294042883,985724156,388968179,819754989,271627185,378316864,820060916,436058499,385836880,818060440,727928431,737435034,888699172,961120185,907997012,619204728,804452206,108201344,986517084,650443054], 95) == 145586000
        return


if __name__ == '__main__':
    unittest.main()
