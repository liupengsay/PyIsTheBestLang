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

mod = 10 ** 9 + 7

pre = ["abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
ind = {}
for i in range(len(pre)):
    for w in range(len(pre[i])):
        ind[pre[i][w]] = str(i+2)*(w+1)
print(ind)