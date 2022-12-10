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
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy

import bisect

import random
import math

from sortedcontainers import SortedList


# int(input().strip())
# [int(w) for w in input().split() if w]
# [float(w) for w in input().split() if w]
# sys.setrecursionlimit(10000000)
#n, c = [int(w) for w in input().strip().split() if w]
import numpy as np
import math
import bisect
from functools import lru_cache
from collections import defaultdict
import bisect

import math
import heapq
import sys
from collections import defaultdict, Counter, deque
from functools import lru_cache
input = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(str(x)+'\n')
sys.setrecursionlimit(10000000)

import math

s = input().strip()




def longestPalindrome(s: str) -> str:
    n = len(s)
    ans = ""
    for i in range(n):
        for x, y in [[i, i],[i,i+1]]:
            while x>=0 and y<n and s[x]==s[y]:
                if y-x+1>len(ans):
                    ans=s[x:y+1]
                x-=1
                y+=1
    return ans


print(longestPalindrome(s))