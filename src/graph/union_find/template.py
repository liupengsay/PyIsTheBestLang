from collections import defaultdict

from math import inf
from typing import DefaultDict, List


class UnionFind:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n
        return

    def find(self, x):
        lst = []
        while x != self.root[x]:
            lst.append(x)
            # merge to the direct root node after query
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] >= self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # assign the rank of non-root nodes to 0
        self.size[root_x] = 0
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        # get the nodes list of every root
        part = defaultdict(list)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self) -> DefaultDict[int, int]:
        # get the size of every root
        size = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
        return size


class UnionFindRightRoot:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        return

    def find(self, x):
        lst = []
        while x != self.root[x]:
            lst.append(x)
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        # select the bigger node as root
        if root_x > root_y:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        return True


class UnionFindWeighted:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n
        self.front = [0] * n
        return

    def find(self, x):
        lst = []
        while x != self.root[x]:
            lst.append(x)
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        lst.append(x)
        m = len(lst)
        for i in range(m - 2, -1, -1):
            self.front[lst[i]] += self.front[lst[i + 1]]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        self.front[root_x] += self.size[root_y]
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        self.size[root_x] = 0
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)


class PersistentUnionFind:
    def __init__(self, n):
        self.rank = [0] * n
        self.root = list(range(n))
        self.version = [inf] * n

    def union(self, x, y, tm):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.version[root_y] = tm
                self.root[root_y] = root_x
            else:
                self.version[root_x] = tm
                self.root[root_x] = root_y
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1
            return True
        return False

    def find(self, x, tm=inf):
        if x == self.root[x] or self.version[x] >= tm:
            return x
        return self.find(self.root[x], tm)

    def is_connected(self, x, y, tm):
        return self.find(x, tm) == self.find(y, tm)


class UnionFindLeftRoot:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.part = n
        return

    def find(self, x):
        lst = []
        while x != self.root[x]:
            lst.append(x)
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        # select the smaller node as root
        if root_x <= root_y:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        return True
