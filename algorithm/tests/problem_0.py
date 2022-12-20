
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


class FastIO:
    def __init__(self):
        return

    @staticmethod
    def _read():
        return sys.stdin.readline().strip()

    def read_int(self):
        return int(self._read())

    def read_float(self):
        return int(self._read())

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

    def read_strs(self):
        return self._read().split()

    def read_list_str(self):
        return self._read().split()

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')


def find_floor(fun, left, right):
    error = 1e-2
    while left < right - error:
        diff = (right - left) / 3
        mid1 = left + diff
        mid2 = left + 2 * diff
        dist1 = fun(mid1)
        dist2 = fun(mid2)
        if dist1 < dist2:
            right = mid2
        elif dist1 > dist2:
            left = mid1
        else:
            left = mid1
            right = mid2
    return left if fun(left) < fun(right) else right


def main(ac=FastIO()):
    n = ac.read_int()
    pos_x = []
    pos_y = []
    for _ in range(n):
        x, y = ac.read_ints()
        pos_x.append(x)
        pos_y.append(y)
    pos_x.sort()
    pos_y.sort()

    def fun_x(a):
        res = 0
        for i in range(n):
            res += abs(pos_x[i]-a-i)
        return res

    def fun_y(b):
        return sum(abs(y-b) for y in pos_y)

    floor_x = find_floor(fun_x, pos_x[0]-n, pos_x[-1]+n)
    floor_y = find_floor(fun_y, pos_y[0]-n, pos_y[-1]+n)

    ans = float("inf")
    for a in range(-1, 2):
        for b in range(-2, 2):
            x = int(floor_x) + a
            y = int(floor_y) + b
            ans = min(ans, fun_x(x) + fun_y(y))
    ac.st(ans)
    return


main()
