from collections import defaultdict

from src.utils.fast_io import inf


class UnionFind:
    def __init__(self, n: int) -> None:
        self.root_or_size = [-1] * n
        self.part = n
        return

    def find(self, x):
        y = x
        while self.root_or_size[x] >= 0:
            # range_merge_to_disjoint to the direct root node after query
            x = self.root_or_size[x]
        while y != x:
            self.root_or_size[y], y = x, self.root_or_size[y]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.root_or_size[root_x] < self.root_or_size[root_y]:
            root_x, root_y = root_y, root_x
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
        self.part -= 1
        return True

    def union_left(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        self.root_or_size[root_x] += self.root_or_size[root_y]
        self.root_or_size[root_y] = root_x
        self.part -= 1
        return True

    def union_right(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_node_size(self, x):
        return -self.root_or_size[self.find(x)]

    def get_root_part(self):
        # get the nodes list of every root
        part = defaultdict(list)
        n = len(self.root_or_size)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        # get the size of every root
        size = defaultdict(int)
        n = len(self.root_or_size)
        for i in range(n):
            if self.find(i) == i:
                size[i] = -self.root_or_size[i]
        return size


class UnionFindWeighted:
    def __init__(self, n: int) -> None:
        self.root_or_size = [-1] * n
        self.front = [0] * n
        return

    def find(self, x):
        lst = []
        while self.root_or_size[x] >= 0:
            lst.append(x)
            x = self.root_or_size[x]
        for w in lst:
            self.root_or_size[w] = x
        lst.append(x)
        m = len(lst)
        for i in range(m - 2, -1, -1):
            self.front[lst[i]] += self.front[lst[i + 1]]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        self.front[root_x] -= self.root_or_size[root_y]
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
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

