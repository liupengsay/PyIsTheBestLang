from collections import defaultdict

from src.utils.fast_io import inf


class UnionFind:
    def __init__(self, n: int) -> None:
        self.root_or_size = [-1] * n
        self.part = n
        self.n = n
        return

    def initialize(self):
        for i in range(self.n):
            self.root_or_size[i] = -1
        self.part = self.n
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

    def union_max(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if root_x > root_y:
            root_x, root_y = root_y, root_x
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
        self.part -= 1
        return

    def union_min(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if root_x < root_y:
            root_x, root_y = root_y, root_x
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
        self.part -= 1
        return

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def size(self, x):
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


class UnionFindGeneral:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n
        return

    def find(self, x):
        y = x
        while x != self.root[x]:
            # range_merge_to_disjoint to the direct root node after query
            x = self.root[x]
        while y != x:
            self.root[y], y = x, self.root[y]
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
        self.size[root_x] = 0
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        part = defaultdict(list)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        size = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
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
        while not (x == self.root[x] or self.version[x] >= tm):
            x = self.root[x]
        return x

    def is_connected(self, x, y, tm):
        return self.find(x, tm) == self.find(y, tm)



class UnionFindSP:
    def __init__(self, n: int) -> None:
        self.root = list(range(n))
        self.size = [0] * n
        self.height = [0] * n
        self.n = n
        return

    def find(self, x):
        y = x
        while x != self.root[x]:
            # range_merge_to_disjoint to the direct root node after query
            x = self.root[x]
        while y != x:
            self.root[y], y = x, self.root[y]
        return x

    def union_right(self, x, y, h):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return 0
        if root_x > root_y:
            root_x, root_y = root_y, root_x
        self.size[root_y] += self.size[root_x]
        self.height[root_y] += self.height[root_x]
        self.root[root_x] = root_y
        if root_y == self.n - 1:
            return h * self.size[root_x] - self.height[root_x]
        return 0


class UnionFindInd:
    def __init__(self, n: int, cnt) -> None:
        self.root_or_size = [-1] * n * cnt
        self.part = [n] * cnt
        self.n = n
        return

    def initialize(self, ind):
        for i in range(self.n):
            self.root_or_size[ind * self.n + i] = -1
        self.part[ind] = self.n
        return

    def find(self, x, ind):
        y = x
        while self.root_or_size[ind * self.n + x] >= 0:
            # range_merge_to_disjoint to the direct root node after query
            x = self.root_or_size[ind * self.n + x]
        while y != x:
            self.root_or_size[ind * self.n + y], y = x, self.root_or_size[ind * self.n + y]
        return x

    def union(self, x, y, ind):
        root_x = self.find(x, ind)
        root_y = self.find(y, ind)
        if root_x == root_y:
            return False
        if self.root_or_size[ind * self.n + root_x] < self.root_or_size[ind * self.n + root_y]:
            root_x, root_y = root_y, root_x
        self.root_or_size[ind * self.n + root_y] += self.root_or_size[ind * self.n + root_x]
        self.root_or_size[ind * self.n + root_x] = root_y
        self.part[ind] -= 1
        return True

    def is_connected(self, x, y, ind):
        return self.find(x, ind) == self.find(y, ind)

