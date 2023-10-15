from math import inf
from typing import List


class PointAddRangeSum:
    # 模板：树状数组 单点增减 查询前缀和与区间和
    def __init__(self, n: int) -> None:
        # 索引从 1 到 n
        self.n = n
        self.t = [0] * (self.n + 1)  # 默认nums=[0]*n
        # 树状数组中每个位置保存的是其向前 low_bit 的区间和
        return

    @staticmethod
    def _lowest_bit(i: int) -> int:
        # 经典 low_bit 即最后一位二进制为 1 所表示的数
        return i & (-i)

    def build(self, nums: List[int]) -> None:
        # 索引从 1 开始使用数组初始化树状数组
        assert len(nums) == self.n
        pre = [0] * (self.n + 1)
        for i in range(self.n):
            pre[i + 1] = pre[i] + nums[i]
            self.t[i + 1] = pre[i + 1] - pre[i + 1 - self._lowest_bit(i + 1)]
        return

    def point_add(self, i: int, mi: int) -> None:
        # 索引从 1 开始，索引 i 的值增加 mi 且 mi 可正可负
        assert 1 <= i <= self.n
        while i < len(self.t):
            self.t[i] += mi
            i += self._lowest_bit(i)
        return

    def get(self) -> List[int]:
        # 索引从 1 开始使用数组初始化树状数组
        nums = [self.pre_sum(i) for i in range(1, self.n + 1)]
        for i in range(self.n - 1, 0, -1):
            nums[i] -= nums[i - 1]
        return nums

    def pre_sum(self, i: int) -> int:
        # 索引从 1 开始，查询 1 到 i 的前缀区间和
        assert 1 <= i <= self.n
        mi = 0
        while i:
            mi += self.t[i]
            i -= self._lowest_bit(i)
        return mi

    def range_sum(self, x: int, y: int) -> int:
        # 索引从 1 开始，查询 x 到 y 的值
        assert 1 <= x <= y <= self.n
        res = self.pre_sum(y) - self.pre_sum(x - 1) if x > 1 else self.pre_sum(y)
        return res


class RangeAddRangeSum:

    def __init__(self, n: int) -> None:
        self.n = n
        # 索引从 1 开始
        self.t1 = [0] * (n + 1)
        self.t2 = [0] * (n + 1)
        return

    @staticmethod
    def _lowest_bit(x: int) -> int:
        # 经典 low_bit 即最后一位二进制为 1 所表示的数
        return x & (-x)

    def build(self, nums: List[int]) -> None:
        # 索引从 1 开始使用数组初始化树状数组
        assert len(nums) == self.n
        for i in range(self.n):
            self.range_add(i + 1, i + 1, nums[i])
        return

    def get(self) -> List[int]:
        # 索引从 1 开始使用数组初始化树状数组
        nums = [0] * self.n
        for i in range(self.n):
            nums[i] = self.range_sum(i + 1, i + 1)
        return nums

    # 更新单点的差分数值
    def _add(self, k: int, v: int) -> None:
        # 索引从 1 开始将第 k 个数加 v 且 v 可正可负
        v1 = k * v
        while k <= self.n:
            self.t1[k] = self.t1[k] + v
            self.t2[k] = self.t2[k] + v1
            k = k + self._lowest_bit(k)
        return

    # 求差分数组的前缀和
    def _sum(self, t: List[int], k: int) -> int:
        # 索引从 1 开始求前 k 个数的前缀和
        ret = 0
        while k:
            ret = ret + t[k]
            k = k - self._lowest_bit(k)
        return ret

    # 更新差分的区间数值
    def range_add(self, left: int, right: int, v: int) -> None:
        # 索引从 1 开始将区间 [left, right] 的数增加 v 且 v 可正可负
        self._add(left, v)
        self._add(right + 1, -v)
        return

    # 求数组的前缀区间和
    def range_sum(self, left: int, right: int) -> int:
        # 索引从 1 开始查询区间 [left, right] 的和
        a = (right + 1) * self._sum(self.t1, right) - self._sum(self.t2, right)
        b = left * self._sum(self.t1, left - 1) - self._sum(self.t2, left - 1)
        return a - b


class TreeArrayRangeQuerySumXOR:
    # 模板：树状数组 单点增减 查询前缀异或和与区间异或和
    def __init__(self, n: int) -> None:
        # 索引从 1 到 n
        self.t = [0] * (n + 1)
        # 树状数组中每个位置保存的是其向前 low_bit 的区间异或和
        return

    def build(self, nums: List[int]) -> None:
        # 索引从 1 开始使用数组初始化树状数组
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] ^ nums[i]
            self.t[i + 1] = pre[i + 1] ^ pre[i + 1 - self._lowest_bit(i + 1)]
        return

    @staticmethod
    def _lowest_bit(i: int) -> int:
        # 经典 low_bit 即最后一位二进制为 1 所表示的数
        return i & (-i)

    def query(self, i: int) -> int:
        # 索引从 1 开始，查询 1 到 i 的前缀区间和
        mi = 0
        while i:
            mi ^= self.t[i]
            i -= self._lowest_bit(i)
        return mi

    def query_range(self, x: int, y: int) -> int:
        # 索引从 1 开始，查询 x 到 y 的值
        return self.query(y) ^ self.query(x - 1)

    def update(self, i: int, mi: int) -> None:
        # 索引从 1 开始，索引 i 的值异或增加 mi 且 mi 可正可负
        while i < len(self.t):
            self.t[i] ^= mi
            i += self._lowest_bit(i)
        return


class TreeArrayRangeQueryPointUpdateMax:
    # 模板：树状数组 单点增加 前缀区间查询 最大值
    def __init__(self, n):
        # 索引从 1 到 n
        self.t = [0] * (n + 1)

    @staticmethod
    def _lowest_bit(i):
        return i & (-i)

    def query(self, i):
        mx = 0
        while i:
            mx = mx if mx > self.t[i] else self.t[i]
            i -= self._lowest_bit(i)
        return mx

    def update(self, i, mx):
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] > mx else mx
            i += self._lowest_bit(i)
        return


class TreeArrayRangeQueryPointUpdateMin:
    # 模板：树状数组 单点减少 前缀区间查询最小值
    def __init__(self, n):
        # 索引从 1 到 n
        self.inf = inf
        self.t = [self.inf] * (n + 1)

    @staticmethod
    def _lowest_bit(i):
        return i & (-i)

    def query(self, i):
        mi = self.inf
        while i:
            mi = mi if mi < self.t[i] else self.t[i]
            i -= self._lowest_bit(i)
        return mi

    def update(self, i, mi):
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] < mi else mi
            i += self._lowest_bit(i)
        return


class TreeArrayPointUpdateRangeMaxMin:

    # 模板：树状数组 单点增加 区间查询最大值 单点减少 区间查询最小值
    def __init__(self, n: int) -> None:
        self.n = n
        # 原始数组
        self.a = [0] * (n + 1)  # 如果是求最小值设置为 [inf]*(n+1) 最大值设置为[-inf]*(n+1)
        self.tree_ceil = [-inf] * (n + 1)  # 初始化也可以设置为[0]*(n+1)
        self.tree_floor = [inf] * (n + 1)  # 初始化也可以设置为[0]*(n+1)
        return

    @staticmethod
    def low_bit(x):
        return x & -x

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def add(self, x, k):
        # 索引从1开始，更新最大值与最小值
        self.a[x] = k
        while x <= self.n:
            self.tree_ceil[x] = self.max(self.tree_ceil[x], k)
            self.tree_floor[x] = self.min(self.tree_floor[x], k)
            x += self.low_bit(x)
        return

    def add_max(self, x, k):
        # 索引从1开始，单点更新最大值
        if self.a[x] >= k:
            return
        self.a[x] = k
        while x <= self.n:
            self.tree_ceil[x] = self.max(self.tree_ceil[x], k)
            self.tree_floor[x] = self.min(self.tree_floor[x], k)
            x += self.low_bit(x)
        return

    def add_min(self, x, k):
        # 索引从1开始，单点更新最小值
        if self.a[x] <= k:
            return
        self.a[x] = k
        while x <= self.n:
            self.tree_ceil[x] = self.max(self.tree_ceil[x], k)
            self.tree_floor[x] = self.min(self.tree_floor[x], k)
            x += self.low_bit(x)
        return

    def find_max(self, left, r):
        # 索引从1开始
        max_val = float('-inf')
        while r >= left:
            if r - self.low_bit(r) >= left - 1:
                max_val = self.max(max_val, self.tree_ceil[r])
                r -= self.low_bit(r)
            else:
                max_val = self.max(max_val, self.a[r])
                r -= 1
        return max_val

    def find_min(self, left, r):
        # 索引从1开始
        min_val = float('inf')
        while r >= left:
            if r - self.low_bit(r) >= left - 1:
                min_val = self.min(min_val, self.tree_floor[r])
                r -= self.low_bit(r)
            else:
                min_val = self.min(min_val, self.a[r])
                r -= 1
        return min_val


class TreeArray2D:
    def __init__(self, m: int, n: int) -> None:
        # 模板：二维树状数组 单点增减 区间和查询
        self.m = m  # 行数
        self.n = n  # 列数
        self.tree = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化树状数组
        return

    def add(self, x: int, y: int, val: int) -> None:
        # 索引从 1 开始， 单点增加 val 到二维数组中坐标为 [x, y] 的值且 val 可正可负
        i = x
        while i <= self.m:
            j = y
            while j <= self.n:
                self.tree[i][j] += val
                j += (j & -j)
            i += (i & -i)
        return

    def _query(self, x: int, y: int) -> int:
        # 索引从 1 开始， 查询二维数组中 [1, 1] 到 [x, y] 的前缀和
        res = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                res += self.tree[i][j]
                j -= (j & -j)
            i -= (i & -i)
        return res

    def range_query(self, x1: int, y1: int, x2: int, y2: int) -> int:
        # 索引从 1 开始， 查询二维数组中 [x1, y1] 到 [x2, y2] 的区间和
        return self._query(x2, y2) - self._query(x2, y1 - 1) - self._query(x1 - 1, y2) + self._query(x1 - 1, y1 - 1)


class TreeArray2DRange:
    def __init__(self, m: int, n: int) -> None:
        # 模板：二维树状数组 区间增减 区间和查询
        self.m = m  # 行数
        self.n = n  # 列数
        self.m = m
        self.n = n
        self.t1 = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化树状数组
        self.t2 = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化树状数组
        self.t3 = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化树状数组
        self.t4 = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化树状数组
        return

    def _add(self, x: int, y: int, val: int) -> None:
        # 索引从 1 开始， 单点增加 val 到二维数组中坐标为 [x, y] 的差分数组值且 val 可正可负
        i = x
        while i <= self.m:
            j = y
            while j <= self.n:
                self.t1[i][j] += val
                self.t2[i][j] += val * x
                self.t3[i][j] += val * y
                self.t4[i][j] += val * x * y
                j += (j & -j)
            i += (i & -i)
        return

    def range_add(self, x1: int, y1: int, x2: int, y2: int, val: int) -> None:
        # 索引从 1 开始， 区间增加 val 到二维数组中坐标为左上角 [x1, y1] 到右下角的 [x2, y2] 且 val 可正可负
        self._add(x1, y1, val)
        self._add(x1, y2 + 1, -val)
        self._add(x2 + 1, y1, -val)
        self._add(x2 + 1, y2 + 1, val)
        return

    def _query(self, x: int, y: int) -> int:
        # 索引从 1 开始， 查询二维数组中 [1, 1] 到 [x, y] 的前缀和
        res = 0
        i = x
        while i:
            j = y
            while j:
                res += (x + 1) * (y + 1) * self.t1[i][j] - (y + 1) * self.t2[i][j] - (x + 1) * self.t3[i][j] + \
                       self.t4[i][j]
                j -= (j & -j)
            i -= (i & -i)
        return res

    def range_query(self, x1: int, y1: int, x2: int, y2: int) -> int:
        # 索引从 1 开始， 查询二维数组中 [x1, y1] 到 [x2, y2] 的区间和
        return self._query(x2, y2) - self._query(x2, y1 - 1) - self._query(x1 - 1, y2) + self._query(x1 - 1, y1 - 1)


class TreeArray2DRangeMaxMin:
    # 模板：树状数组 单点增加区间查询最大值 单点减少区间查询最小值（暂未调通）
    def __init__(self, m: int, n: int) -> None:
        self.m = m
        self.n = n
        self.a = [[0] * (n + 1) for _ in range(m + 1)]
        self.tree_ceil = [[0] * (n + 1) for _ in range(m + 1)]  # 最大值只能持续增加
        self.tree_floor = [[float('inf')] * (n + 1) for _ in range(m + 1)]  # 最小值只能持续减少
        return

    @staticmethod
    def low_bit(x):
        return x & -x

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def add(self, x, y, k):
        # 索引从1开始
        self.a[x][y] = k
        i = x
        while i <= self.m:
            j = y
            while j <= self.n:
                self.tree_ceil[i][j] = self.max(self.tree_ceil[i][j], k)
                self.tree_floor[i][j] = self.min(self.tree_floor[i][j], k)
                j += self.low_bit(j)
            i += self.low_bit(i)
        return

    def find_max(self, x1, y1, x2, y2):
        # 索引从1开始
        max_val = float('-inf')
        i1, i2 = x1, x2
        while i2 >= i1:
            if i2 - self.low_bit(i2) >= i1 - 1:

                #########
                j1, j2 = y1, y2
                while j2 >= j1:
                    if j2 - self.low_bit(j2) >= j1 - 1:
                        max_val = self.max(max_val, self.tree_ceil[i2][j2])
                        j2 -= self.low_bit(j2)
                    else:
                        max_val = self.max(max_val, self.a[i2][j2])
                        j2 -= 1
                ##########

                i2 -= self.low_bit(i2)
            else:

                #########
                j1, j2 = y1, y2
                while j2 >= j1:
                    if j2 - self.low_bit(j2) >= j1 - 1:
                        max_val = self.max(max_val, self.tree_ceil[i2][j2])
                        j2 -= self.low_bit(j2)
                    else:
                        max_val = self.max(max_val, self.a[i2][j2])
                        j2 -= 1
                ##########
                max_val = self.max(max_val, max(self.a[i2][y1:y2 + 1]))
                i2 -= 1
        return max_val

