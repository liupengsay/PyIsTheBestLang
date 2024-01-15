import random
from typing import List


class StringHash:
    def __init__(self, lst: List[int]):
        """two mod to avoid hash crush"""
        self.n = len(lst)
        self.p = [random.randint(26, 100), random.randint(26, 100)]
        self.mod = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]
        self.pre = [[0] * (self.n + 1), [0] * (self.n + 1)]
        self.pp = [[1] * (self.n + 1), [1] * (self.n + 1)]
        for j, w in enumerate(lst):
            for i in range(2):
                self.pre[i][j + 1] = (self.pre[i][j] * self.p[i] + w) % self.mod[i]
                self.pp[i][j + 1] = (self.pp[i][j] * self.p[i]) % self.mod[i]
        return

    def query(self, x, y):
        """range hash value index start from 0"""
        # assert 0 <= x <= y <= self.n - 1
        if y < x:
            return 0, 0
        # with length  y - x + 1
        ans = tuple((self.pre[i][y + 1] - self.pre[i][x] * self.pp[i][y - x + 1]) % self.mod[i] for i in range(2))
        return ans
