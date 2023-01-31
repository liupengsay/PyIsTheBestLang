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
from heapq import nlargest
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





def main(ac=FastIO()):
    import io, os
    import collections
    input = io.BytesIO(os.read(0, os.fstat(0).st_size)).readline
    R = lambda: map(int, input().split())

    n, m = R()

    dp = [[inf] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0
    for _ in range(m):
        x, y, w = R()
        x -= 1
        y -= 1
        w = ac.min(dp[x][y], w)
        dp[x][y] = dp[y][x] = w

    for k in range(n):
        for i in range(n):
            for j in range(i+1, n):
                dp[i][j] = dp[j][i] = ac.min(dp[i][j], dp[i][k]+dp[k][j])

    ans = inf
    for i in range(n):
        for j in range(i + 1, n):
            cur = 0
            for a in range(n):
                for b in range(a + 1, n):
                    x = ac.min(dp[a][i] + dp[j][b], dp[a][j]+dp[i][b])
                    cur += min(dp[a][b], x)
            ans = ans if ans < cur else cur
    ac.st(ans)
    return


main()
