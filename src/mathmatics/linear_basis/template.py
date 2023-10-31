class LinearBasis:
    def __init__(self, lst):
        """线性基类由原数组lst生成"""
        self.n = 64
        self.lst = lst
        self.linear_basis = [0] * self.n
        self.gen_linear_basis()
        return

    def gen_linear_basis(self):
        for num in self.lst:
            self.add(num)
        return

    def add(self, num):
        """加入新数字到线性基"""
        for i in range(self.n - 1, -1, -1):
            if num & (1 << i):
                if self.linear_basis[i]:
                    num ^= self.linear_basis[i]
                else:
                    self.linear_basis[i] = num
                    break
        return

    def query_xor(self, num):
        """查询数字是否可以由原数组中数字异或得到"""
        for i in range(self.n, -1, -1):
            if num & (1 << i):
                num ^= self.linear_basis[i]
        return num == 0

    def query_max(self):
        """查询原数组的最大异或和"""
        ans = 0
        for i in range(self.n - 1, -1, -1):
            if ans ^ self.linear_basis[i] > ans:
                ans ^= self.linear_basis[i]
        return ans

    def query_min(self):
        """查询原数组的最小异或和"""
        if 0 in self.lst or self.query_xor(0):
            return 0
        for i in range(0, self.n + 1):
            if self.linear_basis[i]:
                return self.linear_basis[i]
        return 0

    def query_k_rank(self, k):
        """查询原数组异或和的第K小"""
        if self.query_xor(0):
            k -= 1
        ans = 0
        for i in range(self.n - 1, -1, -1):
            if k & (1 << i) and self.linear_basis[i]:
                ans ^= self.linear_basis[i]
                k ^= (1 << i)
        return ans if not k else -1

    def query_k_smallest(self, num):
        """查询数字是原数组中数字异或的第K小"""
        if num == 0:
            return 1 if self.query_xor(0) else -1
        ans = 0
        for i in range(self.n - 1, -1, -1):
            if num & (self.linear_basis[i]):
                ans ^= (1 << i)
                num ^= self.linear_basis[i]
        return ans + 1 if not num else -1
