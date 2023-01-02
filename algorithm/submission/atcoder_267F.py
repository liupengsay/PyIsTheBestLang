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

sys.setrecursionlimit(10000000)

# https://atcoder.jp/contests/abc267/submissions/me
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

    @staticmethod
    def bootstrap(f, stack=[]):
        def wrappedfunc(*args, **kwargs):
            if stack:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if type(to) is GeneratorType:
                        stack.append(to)
                        to = next(to)
                    else:
                        stack.pop()
                        if not stack:
                            break
                        to = stack[-1].send(to)
                return to
        return wrappedfunc


class TreeDiameter:
    # 任取树中的一个节点x，找出距离它最远的点y，那么点y就是这棵树中一条直径的一个端点。我们再从y出发，找出距离y最远的点就找到了一条直径。
    # 这个算法依赖于一个性质：对于树中的任一个点，距离它最远的点一定是树上一条直径的一个端点。
    def __init__(self, edge):
        self.edge = edge
        self.n = len(self.edge)
        return

    def get_farest(self, node):
        q = deque([(node, -1)])
        while q:
            node, pre = q.popleft()
            for x in self.edge[node]:
                if x != pre:
                    q.append((x, node))
        return node

    def get_diameter_node(self):
        # 获取树的直径端点
        x = self.get_farest(0)
        y = self.get_farest(x)
        return x, y


def main(ac=FastIO()):
    n = ac.read_int()
    edge = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = ac.read_ints_minus_one()
        edge[u].append(v)
        edge[v].append(u)
    # 获取任意直径的端点
    root1, root2 = TreeDiameter(edge).get_diameter_node()
    q = ac.read_int()
    dct = [set() for _ in range(n)]

    # 读取答案
    ans = [-1]*q
    for i in range(q):
        u, k = ac.read_ints()
        u -= 1
        dct[u].add((i, k))

    @ac.bootstrap
    def dfs(node, fa):
        path.append(node)
        for x, z in list(dct[node]):
            if len(path) >= z+1:
                ans[x] = path[-z-1] + 1
                dct[node].discard((x, z))
        for y in edge[node]:
            if y != fa:
                yield dfs(y, node)
        path.pop()
        yield

    path = []
    dfs(root1, -1)
    dfs(root2, -1)
    for a in ans:
        ac.st(a)
    return


main()
