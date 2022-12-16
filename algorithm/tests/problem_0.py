
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
    def read():
        return sys.stdin.readline().strip()

    def read_int(self):
        return int(self.read())

    def read_ints(self):
        return map(int, self.read().split())

    def read_ints_minus_one(self):
        return map(lambda x: int(x) - 1, self.read().split())

    def read_list_ints(self):
        return list(map(int, self.read().split()))

    def read_list_ints_minus_one(self):
        return list(map(lambda x: int(x) - 1, self.read().split()))

    def read_str(self):
        return self.read()

    def read_list_str(self):
        return self.read().split()

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')


def main(ac=FastIO()):
    m, n, x, y = ac.read_ints()
    cnt = [[0] * n for _ in range(m)]
    last = [[-1] * n for _ in range(m)]
    for k in range(1, x + 1):
        x1, y1, x2, y2 = ac.read_ints_minus_one()
        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                cnt[i][j] += 1
                last[i][j] = k
    for _ in range(y):
        a, b = ac.read_ints_minus_one()
        if cnt[a][b]:
            ac.lst(["Y", cnt[a][b], last[a][b]])
        else:
            ac.st("N")
    return


main()
