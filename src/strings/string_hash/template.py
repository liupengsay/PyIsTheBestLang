import random


class StringHash:
    # 注意哈希碰撞，需要取两个质数与模进行区分
    def __init__(self, n, s):
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
        # 模板：字符串区间的哈希值，索引从 0 开始
        ans = [0, 0]
        for i in range(2):
            if x <= y:
                ans[i] = (self.pre[i][y + 1] - self.pre[i][x] * self.pp[i][y - x + 1]) % self.mod[i]
        return ans
