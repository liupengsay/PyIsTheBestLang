

import bisect
import random

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations, permutations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
import heapq
from functools import lru_cache


q = int(input().strip())
lst = []
for _ in range(q):
    op, x = [int(w) for w in input().strip().split() if w]
    if op == 1:
        ans = 0
        for num in lst:
            if num < x:
                ans += 1
            else:
                break
        print(ans+1)
    elif op == 2:

        pre = lst[0]
        rank = 1
        for num in lst:
            if num !=pre:
                rank += 1
            pre = num
            if rank == x:
                print(num)
                break
    elif op == 3:
        ans = -2147483647
        for num in lst:
            if num >= x:
                break
            else:
                ans = num
        print(ans)
    elif op == 4:
        ans = 2147483647
        for num in lst[::-1]:
            if num <= x:
                break
            else:
                ans = num
        print(ans)
    else:
        lst.append(x)
        lst.sort()