import bisect
import random
import re
import sys
import unittest
from typing import List, Callable
from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache, cmp_to_key
from itertools import combinations, accumulate, chain
from sortedcontainers import SortedList, SortedDict, SortedSet
from sortedcontainers import SortedDict
from functools import reduce
from heapq import heappush, heappop, heappushpop, heapify
from operator import xor, mul, add
from functools import lru_cache
from math import inf
import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy
from sortedcontainers import SortedList

# sys.set_int_max_str_digits(0)  # 大数的范围坑


def ac_max(a, b):
    return a if a > b else b


def ac_min(a, b):
    return a if a < b else b


class Solution:
    def longestRepeating(self, s: str, queryCharacters: str, queryIndices: List[int]) -> List[int]:
        lst = SortedList()
        s = list(s)
        n = len(s)
        pre = s[0]
        cnt = 1
        length = SortedList()
        for i in range(1, n):
            if s[i] == pre:
                cnt += 1
            else:
                lst.add([i - cnt, i - 1])
                length.add(cnt)
                cnt = 1
                pre = s[i]
        lst.add([n - cnt, n - 1])
        length.add(cnt)
        ans = []

        def find(target_ind):
            ii = lst.bisect_left([target_ind, target_ind])
            for xx in [ii - 1, ii, ii + 1]:
                if 0 <= xx < len(lst) and lst[xx][0] <= target_ind <= lst[xx][1]:
                    return xx

        for ind, w in zip(queryIndices, queryCharacters):
            if s[ind] == w:
                ans.append(length[-1])
                continue
            s[ind] = w

            j = find(ind)
            x, y = lst[j]
            if x < ind < y:
                lst.pop(j)
                length.discard(y - x + 1)

                lst.add([ind, ind])
                length.add(1)

                lst.add([x, ind - 1])
                length.add(ind - x)

                lst.add([ind + 1, y])
                length.add(y - ind)

            elif x == ind < y:
                lst.pop(j)
                length.discard(y - x + 1)

                lst.add([ind + 1, y])
                length.add(y - ind)

                if not x or s[x - 1] != s[x]:
                    lst.add([x, x])
                    length.add(1)
                else:
                    j = find(x-1)
                    a, b = lst.pop(j)
                    length.discard(b - a + 1)

                    lst.add([a, ind])
                    length.add(ind - a + 1)

            elif x < y == ind:
                lst.pop(j)
                length.discard(y - x + 1)
                lst.add([x, ind - 1])
                length.add(ind - x)
                if ind == n - 1 or s[ind + 1] != s[ind]:
                    lst.add([ind, ind])
                    length.add(1)
                else:
                    j = find(ind+1)
                    a, b = lst.pop(j)
                    length.discard(b - a + 1)

                    lst.add([ind, b])
                    length.add(b - ind + 1)

            else:  # x=y=ind
                lst.pop(j)
                length.discard(1)

                if not x or s[x - 1] != s[ind]:
                    lst.add([x, x])
                    length.add(1)
                    a, b = x, x
                else:
                    j = find(x-1)
                    a, b = lst.pop(j)
                    length.discard(b - a + 1)

                    lst.add([a, ind])
                    length.add(ind - a + 1)
                    a, b = a, ind

                if not (ind == n - 1 or s[ind + 1] != s[ind]):
                    j = find(ind + 1)
                    c, d = lst.pop(j)
                    length.discard(d - c + 1)

                    lst.discard([a, b])
                    length.discard(b - a + 1)

                    lst.add([a, d])
                    length.add(d - a + 1)

            ans.append(length[-1])

        return ans







Solution().longestRepeating("ooulnzhezwtlpqshzwalbcbkajxbleablfcfipgfmmpyqyjefrkkdynnldtorwzbkpwzrfxhyocecbml"
,"ustethyhmmvrwxqdyvqtydfeddomypwnlogqsmkmleao"
,[1,44,15,73,64,68,75,42,62,78,45,38,63,74,51,45,45,36,13,61,64,17,6,78,58,53,44,52,0,45,48,76,40,58,39,36,50,67,24,8,70,35,26,32])
