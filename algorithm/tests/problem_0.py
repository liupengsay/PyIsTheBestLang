from __future__ import division
import copy
import random
import heapq
import math
import sys
import bisect
import re
import time
import datetime
from functools import lru_cache
from collections import deque
from collections import Counter
from collections import defaultdict
from itertools import combinations
from itertools import permutations
from itertools import accumulate
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



class SegmentTreeRangeUpdateXORSum:
    def __init__(self):
        # 区间值01翻转与区间和查询
        self.cover = defaultdict(int)
        self.lazy = defaultdict(int)

    def push_down(self, i, s, m, t):
        if self.lazy[i]:
            self.cover[2 * i] = m - s + 1 - self.cover[2*i]
            self.cover[2 * i + 1] = t - m - self.cover[2 * i + 1]

            self.lazy[2 * i] ^= self.lazy[i]  # 注意使用异或抵消查询
            self.lazy[2 * i + 1] ^= self.lazy[i]  # 注意使用异或抵消查询

            self.lazy[i] = 0

    def update(self, left, r, s, t, val, i):
        if left <= s and t <= r:
            self.cover[i] = t-s+1 - self.cover[i]
            self.lazy[i] ^= val  # 注意使用异或抵消查询
            return
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        if left <= m:
            self.update(left, r, s, m, val, 2 * i)
        if r > m:
            self.update(left, r, m + 1, t, val, 2 * i + 1)
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query(self, left, r, s, t, i):
        if left <= s and t <= r:
            return self.cover[i]
        m = s + (t - s) // 2
        self.push_down(i, s, m, t)
        ans = 0
        if left <= m:
            ans += self.query(left, r, s, m, 2 * i)
        if r > m:
            ans += self.query(left, r, m + 1, t, 2 * i + 1)
        return ans


def main(ac=FastIO()):
    n, m = ac.read_ints()
    tree = SegmentTreeRangeUpdateXORSum()
    s = ac.read_str()
    cnt = 0
    for i in range(n):
        if s[i] == "1":
            cnt += 1
        else:
            if cnt:
                tree.update(i+1-cnt, i, 1, n, 1, 1)
            cnt = 0
    if cnt:
        tree.update(n-cnt+1, n, 1, n, 1, 1)
    for _ in range(m):
        op, left ,right = ac.read_ints()
        if not op:
            tree.update(left, right, 1, n, 1, 1)
        else:
            ac.st(tree.query(left, right, 1, n, 1))
    return


main()
