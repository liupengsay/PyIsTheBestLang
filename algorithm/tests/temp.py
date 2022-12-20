import random
import heapq
import math
import sys
import bisect
from functools import lru_cache
from collections import Counter
from collections import defaultdict
from itertools import combinations
from itertools import permutations

sys.setrecursionlimit(10000000)


def n_queue(x):
    res = 0

    def dfs(i):
        nonlocal res
        if i == x:
            res += 1
            return
        for j in range(x):
            if j not in col:
                col.add(j)
                dfs(i+1)
                col.discard(j)
        return

    col = set()
    dfs(0)
    return res


result = dict()

for k in range(1, 21):
    print(k)
    result[k] = n_queue(k)

print(result)
