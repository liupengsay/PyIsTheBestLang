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


class FastIO:
    def __init__(self):
        return

    @staticmethod
    def _read():
        return sys.stdin.readline().strip()

    def read_int(self):
        return int(self._read())

    def read_float(self):
        return float(self._read())

    def read_ints(self):
        return map(int, self._read().split())

    def read_floats(self):
        return map(float, self._read().split())

    def read_ints_minus_one(self):
        return map(lambda x: int(x) - 1, self._read().split())

    def read_list_ints(self):
        return list(map(int, self._read().split()))

    def read_list_floats(self):
        return list(map(float, self._read().split()))

    def read_list_ints_minus_one(self):
        return list(map(lambda x: int(x) - 1, self._read().split()))

    def read_str(self):
        return self._read()

    def read_list_strs(self):
        return self._read().split()

    def read_list_str(self):
        return list(self._read())

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')

    @staticmethod
    def round_5(f):
        res = int(f)
        if f - res >= 0.5:
            res += 1
        return res

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def bootstrap(f, stack=[]):
        def wrappedfunc(*args, **kwargs):
            if stack:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if isinstance(to, GeneratorType):
                        stack.append(to)
                        to = next(to)
                    else:
                        stack.pop()
                        if not stack:
                            break
                        to = stack[-1].send(to)
                return to
        return wrappedfunc


def check1(nums, n ,c):
    stack = []
    ans = []

    queue = deque()
    right = 0
    for i in range(n):
        m = len(stack)

        # 维护右边还可以在栈长度范围内增加元素的最小值
        while queue and queue[0][1] < i:
            queue.popleft()
        for x in range(right, min(right + c - m, n, i + c - m)):
            while queue and queue[-1][0] >= nums[x]:
                queue.pop()
            queue.append([nums[x], x])
        right = min(right + c - m, n, i + c - m)

        # 如果栈满了或者右边没有更小于的直接出队
        while (queue and stack and stack[-1] <= queue[0][0]) or len(stack) == c:
            ans.append(stack.pop())
        stack.append(nums[i])
    ans.extend(stack[::-1])
    return ans


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


def main(ac=FastIO()):
    n, c = ac.read_ints()

    nums = ac.read_list_ints()

    ac.lst(check3(nums, n, c))
    return


main()
