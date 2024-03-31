from src.utils.fast_io import inf


class PointAddRangeSum:
    def __init__(self, n: int, initial=0) -> None:
        """index from 1 to n"""
        self.n = n
        self.t = [initial] * (self.n + 1)  # default nums = [0]*n
        return

    @staticmethod
    def _lowest_bit(i: int) -> int:
        return i & (-i)

    def _pre_sum(self, i: int) -> int:
        """index start from 1 and the prefix sum of nums[:i] which is 0-index"""

        val = 0  # assert 1 <= i <= self.n
        while i:
            val += self.t[i]
            i -= self._lowest_bit(i)
        return val

    def build(self, nums) -> None:
        """initialize the tree array"""
        pre = [0] * (self.n + 1)  # assert len(nums) == self.n
        for i in range(self.n):
            pre[i + 1] = pre[i] + nums[i]
            # meaning of self.t[i+1]
            self.t[i + 1] = pre[i + 1] - pre[i + 1 - self._lowest_bit(i + 1)]
        return

    def get(self):
        """get the original nums sometimes for debug"""
        nums = [self._pre_sum(i) for i in range(1, self.n + 1)]
        for i in range(self.n - 1, 0, -1):
            nums[i] -= nums[i - 1]
        return nums

    def point_add(self, i: int, val: int) -> None:
        """index start from 1 and the value val can be any inter including positive and negative number"""
        while i < len(self.t):  # assert 1 <= i <= self.n
            self.t[i] += val
            i += self._lowest_bit(i)
        return

    def range_sum(self, x: int, y: int) -> int:
        """index start from 1 and the range sum of nums[x-1:y]  which is 0-index"""
        res = self._pre_sum(y) - self._pre_sum(x - 1) if x > 1 else self._pre_sum(y)  # assert 1 <= x <= y <= self.n
        return res

    def bisect_right(self, w):
        # all value in nums must be non-negative
        x, k = 0, 1
        while k * 2 <= self.n:
            k *= 2
        while k > 0:
            if x + k <= self.n and self.t[x + k] <= w:
                w -= self.t[x + k]
                x += k
            k //= 2
        return x


class PointChangeRangeSum:
    def __init__(self, n: int) -> None:
        # index from 1 to n
        self.n = n
        self.t = [0] * (self.n + 1)  # default nums = [0]*n
        return

    @staticmethod
    def _lowest_bit(i: int) -> int:
        return i & (-i)

    def _pre_sum(self, i: int) -> int:
        # index start from 1 and the prefix sum of nums[:i] which is 0-index
        assert 1 <= i <= self.n
        val = 0
        while i:
            val += self.t[i]
            i -= self._lowest_bit(i)
        return val

    def build(self, nums) -> None:
        # initialize
        assert len(nums) == self.n
        pre = [0] * (self.n + 1)
        for i in range(self.n):
            pre[i + 1] = pre[i] + nums[i]
            # meaning of self.t[i+1]
            self.t[i + 1] = pre[i + 1] - pre[i + 1 - self._lowest_bit(i + 1)]
        return

    def point_change(self, i: int, val: int) -> None:
        # index start from 1 and the value val can be any inter including positive and negative number
        assert 1 <= i <= self.n
        pre = self.range_sum(i, i)
        gap = val - pre
        if gap:
            while i < len(self.t):
                self.t[i] += gap
                i += self._lowest_bit(i)
        return

    def get(self):
        # get the original nums sometimes for debug
        nums = [self._pre_sum(i) for i in range(1, self.n + 1)]
        for i in range(self.n - 1, 0, -1):
            nums[i] -= nums[i - 1]
        return nums

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
        # index start from 1 and query the sum of matrix sum(s[y1:y2+1] for s in grid[x1: x2+1])  which is 1-index
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

    def build(self, nums) -> None:
        assert len(nums) == self.n
        for i in range(self.n):
            self.range_add(i + 1, i + 1, nums[i])
        return

    def get(self):
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

    def _sum(self, t, k: int) -> int:
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
        val = 0
        while i:
            val ^= self.t[i]
            i -= self._lowest_bit(i)
        return val

    def build(self, nums) -> None:
        assert len(nums) == self.n
        pre = [0] * (self.n + 1)
        for i in range(self.n):
            pre[i + 1] = pre[i] ^ nums[i]
            self.t[i + 1] = pre[i + 1] ^ pre[i + 1 - self._lowest_bit(i + 1)]
        return

    def get(self):
        return [self.range_xor(i + 1, i + 1) for i in range(self.n)]

    def point_xor(self, i: int, val: int) -> None:
        assert 1 <= i <= self.n
        while i <= self.n:
            self.t[i] ^= val
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

    def point_ascend(self, x, k):
        assert 1 <= x <= self.n
        if self.a[x] >= k:
            return
        self.a[x] = k
        while x <= self.n:
            self.t[x] = max(self.t[x], k)
            x += self._lowest_bit(x)
        return

    def range_max(self, left, r):
        assert 1 <= left <= r <= self.n
        max_val = self.initial
        while r >= left:
            if r - self._lowest_bit(r) >= left - 1:
                max_val = max(max_val, self.t[r])
                r -= self._lowest_bit(r)
            else:
                max_val = max(max_val, self.a[r])
                r -= 1
        return max_val


class PointDescendPreMin:
    def __init__(self, n, initial=inf):
        self.n = n
        self.initial = initial
        self.t = [self.initial] * (n + 1)

    def initialize(self):
        for i in range(self.n + 1):
            self.t[i] = self.initial
        return

    @staticmethod
    def _lowest_bit(i):
        return i & (-i)

    def pre_min(self, i):
        # assert 1 <= i <= self.n
        val = self.initial
        while i:
            val = val if val < self.t[i] else self.t[i]
            i -= self._lowest_bit(i)
        return val

    def point_descend(self, i, val):
        # assert 1 <= i <= self.n
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] < val else val
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

    def point_descend(self, x, k):
        assert 1 <= x <= self.n
        if self.a[x] <= k:
            return
        self.a[x] = k
        while x <= self.n:
            self.t[x] = min(self.t[x], k)
            x += self._lowest_bit(x)
        return

    def range_min(self, left, r):
        assert 1 <= left <= r <= self.n
        min_val = self.initial
        while r >= left:
            if r - self._lowest_bit(r) >= left - 1:
                min_val = min(min_val, self.t[r])
                r -= self._lowest_bit(r)
            else:
                min_val = min(min_val, self.a[r])
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
        assert 0 <= x <= self.m and 0 <= y <= self.n
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

    def add(self, x, y, k):
        # index start from 1
        self.a[x][y] = k
        i = x
        while i <= self.m:
            j = y
            while j <= self.n:
                self.tree_ceil[i][j] = max(self.tree_ceil[i][j], k)
                self.tree_floor[i][j] = min(self.tree_floor[i][j], k)
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
                        max_val = max(max_val, self.tree_ceil[i2][j2])
                        j2 -= self._lowest_bit(j2)
                    else:
                        max_val = max(max_val, self.a[i2][j2])
                        j2 -= 1
                ##########

                i2 -= self._lowest_bit(i2)
            else:

                #########
                j1, j2 = y1, y2
                while j2 >= j1:
                    if j2 - self._lowest_bit(j2) >= j1 - 1:
                        max_val = max(max_val, self.tree_ceil[i2][j2])
                        j2 -= self._lowest_bit(j2)
                    else:
                        max_val = max(max_val, self.a[i2][j2])
                        j2 -= 1
                ##########
                max_val = max(max_val, max(self.a[i2][y1:y2 + 1]))
                i2 -= 1
        return max_val
