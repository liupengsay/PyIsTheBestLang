
from __future__ import division
import copy
import random
import heapq
import math
import sys
import bisect
import datetime
from functools import lru_cache
from collections import deque
from collections import Counter
from collections import defaultdict
from itertools import combinations
from itertools import permutations
from decimal import Decimal, getcontext, MAX_PREC
from types import GeneratorType
from functools import cmp_to_key

inf = float("inf")
sys.setrecursionlimit(10000000)

getcontext().prec = MAX_PREC



def check2(nums, n, c):
    ans = []
    stack = []
    for i in range(n):
        if not stack:
            stack.append(nums[i])
            continue

        add = False
        while stack:
            flag = False
            for j in range(i, min(i+c-len(stack), n)):
                if nums[j] < stack[-1]:
                    flag = True
                    break
            if flag:
                add = True
                stack.append(nums[i])
                break
            else:
                ans.append(stack.pop())
        if not add:
            stack.append(nums[i])
    ans.extend(stack[::-1])
    return ans


def check3(nums, n, c):
    ans = []
    stack = []
    queue = deque()
    j = 0
    for i in range(n):
        if not stack:
            stack.append(nums[i])
            continue

        while queue and queue[0] < i:
            queue.popleft()
        j = i if j < i else i

        add = False
        while stack:
            while j < n and j-i+1 + len(stack) <= c:
                while queue and nums[queue[-1]] >= nums[j]:
                    queue.pop()
                queue.append(j)
                j += 1
                if nums[queue[0]] < stack[-1]:
                    break

            if queue and stack[-1] > nums[queue[0]]:
                add = True
                stack.append(nums[i])
                break
            else:
                ans.append(stack.pop())
        if not add:
            stack.append(nums[i])
    ans.extend(stack[::-1])
    return ans



for _ in range(10):
    n = random.randint(10, 100)
    c = random.randint(1, n)
    nums = [random.randint(1, 100) for _ in range(n)]
    if not check2(nums, n, c) == check3(nums, n, c):
        print(nums, n, c)
        break
