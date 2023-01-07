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





@lru_cache(None)
def dfs(a, b):
    if a>=1 and b>=2:
        if not dfs(a-1, b-2):
            return True
    if a>=2 and b>=1:
        if not dfs(a-2, b-1):
            return True
    return False


for i in range(1, 10):
    for j in range(1, 10):
        print(i, j, dfs(i, j), (i+j)%2)

