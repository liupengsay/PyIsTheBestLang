import bisect
import random
import re
import unittest

from typing import List, Callable, Dict
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, accumulate
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np
from typing import List, Callable
from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList





class Solution:
    def digitsCount(self, d: int, low: int, high: int) -> int:

        def count_digit(num, d):
            # 模板: 计算 1到 num 内数位 d 出现的个数
            @lru_cache(None)
            def dfs(i, cnt, is_limit, is_num):
                if i == n:
                    if is_num:
                        return cnt
                    return 0
                res = 0
                if not is_num:
                    res += dfs(i + 1, 0, False, False)

                floor = 0 if is_num else 1
                ceil = int(s[i]) if is_limit else 9
                for x in range(floor, ceil + 1):
                    res += dfs(i + 1, cnt + int(x == d), is_limit and ceil == x, True)
                return res

            s = str(num)
            n = len(s)
            ans = dfs(0, 0, True, False)
            dfs.cache_clear()
            return ans

        res = count_digit(high, d) - count_digit(low-1, d)
        return res








assert Solution().kSimilarity("ab", "ba") == 1
