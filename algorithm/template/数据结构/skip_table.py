"""
ST表：静态查询区间的最大值，最大公约数等

P3865 【模板】ST 表
https://www.luogu.com.cn/problem/P3865
"""

import sys
from collections import defaultdict, Counter, deque
from functools import lru_cache
input = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(str(x)+'\n')
sys.setrecursionlimit(10000000)

import math

n, m = map(int, input().split())
lst1 = list(map(int, input().split()))
f = [[0] * 18 for i in range(1 + n)]

for i in range(1, n + 1):
    f[i][0] = lst1[i - 1]
for j in range(1, int(math.log2(n)) + 1):  # 一定不能用ceil，必须用floor 或者int    +1
    for i in range(1, n - (1<<j) + 2):
        a = f[i][j - 1]
        b = f[i +  (1<<(j - 1))][j - 1]
        f[i][j] = a if a > b else b

for i in range(m):
    l, r = map(int, input().split())
    k = int(math.log2(r - l + 1))
    a = f[l][k]
    b = f[r - (1<<k) + 1][k]
    print(a if a > b else b)