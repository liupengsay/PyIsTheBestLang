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
from types import GeneratorType
from functools import cmp_to_key
inf = float("inf")
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


def main(ac=FastIO()):
    n, m = ac.read_ints()
    edge = [[] for _ in range(n)]
    rev = [[] for _ in range(n)]
    for _ in range(m):
        u, v, w = ac.read_ints()
        edge[u-1].append([v-1, w])
        rev[v-1].append([u-1, w])

    n = len(edge)
    dis = [float("inf")] * n
    dis[0] = 0
    while stack:
        d, i = heapq.heappop(stack)
        if dis[i] < d:
            continue
        for j, w in edge[i]:
            dj = w + d
            if dj < dis[j]:
                dis[j] = dj
                heapq.heappush(stack, [dj, j])
    ans = sum(dis)

    n = len(rev)
    dis = [float("inf")] * n
    stack = [[0, 0]]
    dis[0] = 0
    while stack:
        d, i = heapq.heappop(stack)
        if dis[i] < d:
            continue
        for j, w in rev[i]:
            dj = w + d
            if dj < dis[j]:
                dis[j] = dj
                heapq.heappush(stack, [dj, j])
    ans += sum(dis)
    ac.st(ans)
    return


main()
