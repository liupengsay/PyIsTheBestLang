import random


class StringHash:
    def __init__(self, n, s):
        """two mod to avoid hash crush"""
        self.n = n
        self.p = [random.randint(26, 100), random.randint(26, 100)]
        self.mod = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]
        self.pre = [[0] * (n + 1), [0] * (n + 1)]
        self.pp = [[1] * (n + 1), [1] * (n + 1)]
        for j, w in enumerate(s):
            for i in range(2):
                self.pre[i][j + 1] = (self.pre[i][j] * self.p[i] + w) % self.mod[i]
                self.pp[i][j + 1] = (self.pp[i][j] * self.p[i]) % self.mod[i]
        return

    def query(self, x, y):
        """range hash value index start from 0"""
        ans = tuple((self.pre[i][y + 1] - self.pre[i][x] * self.pp[i][y - x + 1]) % self.mod[i] for i in range(2))
        return ans
