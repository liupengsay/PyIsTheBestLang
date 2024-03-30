from src.basis.binary_search.template import BinarySearch


class LinearBasis:
    def __init__(self, m=64):
        self.m = m
        self.basis = [0] * self.m
        self.cnt = self.count_diff_xor()
        self.tot = 1 << self.cnt
        self.num = 0
        self.zero = 0
        self.length = 0
        return

    def minimize(self, x):
        for i in range(self.m):
            if x >> i & 1:
                x ^= self.basis[i]
        return x

    def add(self, x):
        assert x <= (1 << self.m) - 1
        x = self.minimize(x)
        self.num += 1
        if x:
            self.length += 1
        self.zero = int(self.length < self.num)

        for i in range(self.m - 1, -1, -1):
            if x >> i & 1:
                for j in range(self.m):
                    if self.basis[j] >> i & 1:
                        self.basis[j] ^= x
                self.basis[i] = x
                self.cnt = self.count_diff_xor()
                self.tot = 1 << self.cnt
                return True
        return False

    def count_diff_xor(self):
        num = 0
        for i in range(self.m):
            if self.basis[i] > 0:
                num += 1
        return num

    def query_kth_xor(self, x):
        res = 0
        for i in range(self.m):
            if self.basis[i]:
                if x & 1:
                    res ^= self.basis[i]
                x >>= 1
        return res

    def query_xor_kth(self, num):
        bs = BinarySearch()

        def check(x):
            return self.query_kth_xor(x) <= num

        return bs.find_int_right(0, self.tot - 1, check)

    def query_max(self):
        return self.query_kth_xor(self.tot - 1)

    def query_min(self):
        # include empty subset
        return self.query_kth_xor(0)

class LinearBasisVector:
    def __init__(self, m):
        self.basis = [[0] * m for _ in range(m)]
        self.m = m
        return

    def add(self, lst):
        for i in range(self.m):
            if self.basis[i][i] and lst[i]:
                a, b = self.basis[i][i], lst[i]
                self.basis[i] = [x * b for x in self.basis[i]]
                lst = [x * a for x in lst]
                lst = [lst[j] - self.basis[i][j] for j in range(self.m)]
        for j in range(self.m):
            if lst[j]:
                self.basis[j] = lst[:]
                return True
        return False