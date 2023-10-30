from math import inf
from typing import List


class PointAddRangeSum:
    def __init__(self, n: int) -> None:
        # index from 1 to n
        self.n = n
        self.t = [0] * (self.n + 1)  # default nums = [0]*n
        return

    @staticmethod
    def _lowest_bit(i: int) -> int:
        return i & (-i)

    def build(self, nums: List[int]) -> None:
        # initialize
        assert len(nums) == self.n
        pre = [0] * (self.n + 1)
        for i in range(self.n):
            pre[i + 1] = pre[i] + nums[i]
            # meaning of self.t[i+1]
            self.t[i + 1] = pre[i + 1] - pre[i + 1 - self._lowest_bit(i + 1)]
        return

    def point_add(self, i: int, mi: int) -> None:
        # index start from 1 and the value mi can be any inter including positive and negative number
        assert 1 <= i <= self.n
        while i < len(self.t):
            self.t[i] += mi
            i += self._lowest_bit(i)
        return

    def get(self) -> List[int]:
        # get the original nums sometimes for debug
        nums = [self._pre_sum(i) for i in range(1, self.n + 1)]
        for i in range(self.n - 1, 0, -1):
            nums[i] -= nums[i - 1]
        return nums

    def _pre_sum(self, i: int) -> int:
        # index start from 1 and the prefix sum of nums[:i] which is 0-index
        assert 1 <= i <= self.n
        mi = 0
        while i:
            mi += self.t[i]
            i -= self._lowest_bit(i)
        return mi

    def range_sum(self, x: int, y: int) -> int:
        # index start from 1 and the range sum of nums[x-1:y]  which is 0-index
        assert 1 <= x <= y <= self.n
        res = self._pre_sum(y) - self._pre_sum(x - 1) if x > 1 else self._pre_sum(y)
        return res


class PointAddRangeSum2D:
    def __init__(self, m: int, n: int) -> None:
        self.m = m
        self.n = n
        self.tree = [[0] * (n + 1) for _ in range(m + 1)]
        return

    def point_add(self, x: int, y: int, val: int) -> None:
        # index start from 1 and val can be any integer
        i = x
        while i <= self.m:
            j = y
            while j <= self.n:
                self.tree[i][j] += val
                j += (j & -j)
            i += (i & -i)
        return

    def _query(self, x: int, y: int) -> int:
        # index start from 1 and query the sum of prefix matrix sum(s[:y] for s in grid[:x])  which is 0-index
        res = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                res += self.tree[i][j]
                j -= (j & -j)
            i -= (i & -i)
        return res

    def range_sum(self, x1: int, y1: int, x2: int, y2: int) -> int:
        # index start from 1 and query the sum of matrix sum(s[y1-1:y2] for s in grid[x1-1: x2])  which is 0-index
        return self._query(x2, y2) - self._query(x2, y1 - 1) - self._query(x1 - 1, y2) + self._query(x1 - 1, y1 - 1)


class RangeAddRangeSum:

    def __init__(self, n: int) -> None:
        self.n = n
        self.t1 = [0] * (n + 1)
        self.t2 = [0] * (n + 1)
        return

    @staticmethod
    def _lowest_bit(x: int) -> int:
        return x & (-x)

    def build(self, nums: List[int]) -> None:
        assert len(nums) == self.n
        for i in range(self.n):
            self.range_add(i + 1, i + 1, nums[i])
        return

    def get(self) -> List[int]:
        nums = [0] * self.n
        for i in range(self.n):
            nums[i] = self.range_sum(i + 1, i + 1)
        return nums

    def _add(self, k: int, v: int) -> None:
        # start from index 1 and v can be any integer
        v1 = k * v
        while k <= self.n:
            self.t1[k] = self.t1[k] + v
            self.t2[k] = self.t2[k] + v1
            k = k + self._lowest_bit(k)
        return

    def _sum(self, t: List[int], k: int) -> int:
        # index start from 1 and query the sum of prefix k number
        ret = 0
        while k:
            ret = ret + t[k]
            k = k - self._lowest_bit(k)
        return ret

    def range_add(self, left: int, right: int, v: int) -> None:
        # index start from 1 and v van be any integer
        self._add(left, v)
        self._add(right + 1, -v)
        return

    def range_sum(self, left: int, right: int) -> int:
        # index start from 1 and query the sum(nums[left-1: right]) which is 0-index array
        a = (right + 1) * self._sum(self.t1, right) - self._sum(self.t2, right)
        b = left * self._sum(self.t1, left - 1) - self._sum(self.t2, left - 1)
        return a - b


class PointXorRangeXor:
    def __init__(self, n: int) -> None:
        self.n = n
        self.t = [0] * (n + 1)
        return

    @staticmethod
    def _lowest_bit(i: int) -> int:
        return i & (-i)

    def _pre_xor(self, i: int) -> int:
        assert 1 <= i <= self.n
        mi = 0
        while i:
            mi ^= self.t[i]
            i -= self._lowest_bit(i)
        return mi

    def build(self, nums: List[int]) -> None:
        assert len(nums) == self.n
        pre = [0] * (self.n + 1)
        for i in range(self.n):
            pre[i + 1] = pre[i] ^ nums[i]
            self.t[i + 1] = pre[i + 1] ^ pre[i + 1 - self._lowest_bit(i + 1)]
        return

    def get(self) -> List[int]:
        return [self.range_xor(i + 1, i + 1) for i in range(self.n)]

    def point_xor(self, i: int, mi: int) -> None:
        assert 1 <= i <= self.n
        while i <= self.n:
            self.t[i] ^= mi
            i += self._lowest_bit(i)
        return

    def range_xor(self, x: int, y: int) -> int:
        assert 1 <= x <= y <= self.n
        res = self._pre_xor(y) ^ self._pre_xor(x - 1) if x > 1 else self._pre_xor(y)
        return res


class PointAscendPreMax:
    def __init__(self, n, initial=-inf):
        self.n = n
        self.initial = initial
        self.t = [initial] * (n + 1)

    @staticmethod
    def _lowest_bit(i):
        return i & (-i)

    def pre_max(self, i):
        assert 1 <= i <= self.n
        mx = self.initial
        while i:
            mx = mx if mx > self.t[i] else self.t[i]
            i -= self._lowest_bit(i)
        return mx

    def point_ascend(self, i, mx):
        assert 1 <= i <= self.n
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] > mx else mx
            i += self._lowest_bit(i)
        return


class PointAscendRangeMax:
    def __init__(self, n: int, initial=-inf) -> None:
        self.n = n
        self.initial = initial
        self.a = [self.initial] * (n + 1)
        self.t = [self.initial] * (n + 1)
        return

    @staticmethod
    def _lowest_bit(x):
        return x & -x

    @staticmethod
    def max(a, b):
        return a if a > b else b

    def point_ascend(self, x, k):
        assert 1 <= x <= self.n
        if self.a[x] >= k:
            return
        self.a[x] = k
        while x <= self.n:
            self.t[x] = self.max(self.t[x], k)
            x += self._lowest_bit(x)
        return

    def range_max(self, left, r):
        assert 1 <= left <= r <= self.n
        max_val = self.initial
        while r >= left:
            if r - self._lowest_bit(r) >= left - 1:
                max_val = self.max(max_val, self.t[r])
                r -= self._lowest_bit(r)
            else:
                max_val = self.max(max_val, self.a[r])
                r -= 1
        return max_val


class PointDescendPreMin:
    def __init__(self, n, initial=inf):
        self.n = n
        self.initial = initial
        self.t = [self.initial] * (n + 1)

    @staticmethod
    def _lowest_bit(i):
        return i & (-i)

    def pre_min(self, i):
        assert 1 <= i <= self.n
        mi = self.initial
        while i:
            mi = mi if mi < self.t[i] else self.t[i]
            i -= self._lowest_bit(i)
        return mi

    def point_descend(self, i, mi):
        assert 1 <= i <= self.n
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] < mi else mi
            i += self._lowest_bit(i)
        return


class PointDescendRangeMin:
    def __init__(self, n: int, initial=inf) -> None:
        self.n = n
        self.initial = initial
        self.a = [self.initial] * (n + 1)
        self.t = [self.initial] * (n + 1)
        return

    @staticmethod
    def _lowest_bit(x):
        return x & -x

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def point_descend(self, x, k):
        assert 1 <= x <= self.n
        if self.a[x] <= k:
            return
        self.a[x] = k
        while x <= self.n:
            self.t[x] = self.min(self.t[x], k)
            x += self._lowest_bit(x)
        return

    def range_min(self, left, r):
        assert 1 <= left <= r <= self.n
        min_val = self.initial
        while r >= left:
            if r - self._lowest_bit(r) >= left - 1:
                min_val = self.min(min_val, self.t[r])
                r -= self._lowest_bit(r)
            else:
                min_val = self.min(min_val, self.a[r])
                r -= 1
        return min_val


class RangeAddRangeSum2D:
    def __init__(self, m: int, n: int) -> None:
        self.m = m  # row
        self.n = n  # col
        self.m = m
        self.n = n
        self.t1 = [[0] * (n + 1) for _ in range(m + 1)]
        self.t2 = [[0] * (n + 1) for _ in range(m + 1)]
        self.t3 = [[0] * (n + 1) for _ in range(m + 1)]
        self.t4 = [[0] * (n + 1) for _ in range(m + 1)]
        return

    def _add(self, x: int, y: int, val: int) -> None:
        # index start from 1 and single point add val and val cam be any integer
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
        # index start from 1 and left up corner is (x1, y1) and right down corner is (x2, y2) and val can be any integer
        self._add(x1, y1, val)
        self._add(x1, y2 + 1, -val)
        self._add(x2 + 1, y1, -val)
        self._add(x2 + 1, y2 + 1, val)
        return

    def _query(self, x: int, y: int) -> int:
        # index start from 1 and query the sum(sum(g[:y]) for g in grid[:x]) which is 0-index
        assert 1 <= x <= self.m and 1 <= y <= self.n
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
        # index start from 1 and left up corner is (x1, y1) and right down corner is (x2, y2)
        return self._query(x2, y2) - self._query(x2, y1 - 1) - self._query(x1 - 1, y2) + self._query(x1 - 1, y1 - 1)


class PointChangeMaxMin2D:
    # not already for use and there still exist some bug
    def __init__(self, m: int, n: int) -> None:
        self.m = m
        self.n = n
        self.a = [[0] * (n + 1) for _ in range(m + 1)]
        self.tree_ceil = [[0] * (n + 1) for _ in range(m + 1)]  # point keep ascend
        self.tree_floor = [[float('inf')] * (n + 1) for _ in range(m + 1)]  # point keep descend
        return

    @staticmethod
    def _lowest_bit(x):
        return x & -x

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    def add(self, x, y, k):
        # index start from 1
        self.a[x][y] = k
        i = x
        while i <= self.m:
            j = y
            while j <= self.n:
                self.tree_ceil[i][j] = self.max(self.tree_ceil[i][j], k)
                self.tree_floor[i][j] = self.min(self.tree_floor[i][j], k)
                j += self._lowest_bit(j)
            i += self._lowest_bit(i)
        return

    def find_max(self, x1, y1, x2, y2):
        assert 1 <= x1 <= x2 <= self.m and 1 <= y1 <= y2 <= self.n
        max_val = inf
        i1, i2 = x1, x2
        while i2 >= i1:
            if i2 - self._lowest_bit(i2) >= i1 - 1:

                #########
                j1, j2 = y1, y2
                while j2 >= j1:
                    if j2 - self._lowest_bit(j2) >= j1 - 1:
                        max_val = self.max(max_val, self.tree_ceil[i2][j2])
                        j2 -= self._lowest_bit(j2)
                    else:
                        max_val = self.max(max_val, self.a[i2][j2])
                        j2 -= 1
                ##########

                i2 -= self._lowest_bit(i2)
            else:

                #########
                j1, j2 = y1, y2
                while j2 >= j1:
                    if j2 - self._lowest_bit(j2) >= j1 - 1:
                        max_val = self.max(max_val, self.tree_ceil[i2][j2])
                        j2 -= self._lowest_bit(j2)
                    else:
                        max_val = self.max(max_val, self.a[i2][j2])
                        j2 -= 1
                ##########
                max_val = self.max(max_val, max(self.a[i2][y1:y2 + 1]))
                i2 -= 1
        return max_val
