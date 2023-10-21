import random


class StringHash:
    def __init__(self, n, s):
        """two mod to avoid hash crush"""
        self.n = n
        self.p = [random.randint(26, 100), random.randint(26, 100)]
        self.mod = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]
        self.pre = [[0], [0]]
        self.pp = [[1], [1]]
        for w in s:
            for i in range(2):
                self.pre[i].append((self.pre[i][-1] * self.p[i] + ord(w) - ord("a")) % self.mod[i])
                self.pp[i].append((self.pp[i][-1] * self.p[i]) % self.mod[i])
        return

    def query(self, x, y):
        """range hash value index start from 0"""
        ans = [0, 0]
        for i in range(2):
            if x <= y:
                ans[i] = (self.pre[i][y + 1] - self.pre[i][x] * self.pp[i][y - x + 1]) % self.mod[i]
        return ans
