import math
import random

from src.utils.fast_io import inf


class MatrixHashReverse:
    def __init__(self, m, n, grid):
        """
        primes = PrimeSieve().eratosthenes_sieve(100)
        primes = [x for x in primes if 26 < x < 100]
        """
        primes = [29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        self.m, self.n = m, n

        self.p1 = primes[random.randint(0, len(primes) - 1)]
        while True:
            self.p2 = primes[random.randint(0, len(primes) - 1)]
            if self.p2 != self.p1:
                break

        ceil = self.m if self.m > self.n else self.n
        self.pp1 = [1] * (ceil + 1)
        self.pp2 = [1] * (ceil + 1)
        self.mod = random.randint(10 ** 9 + 7, (1 << 31) - 1)

        for i in range(1, ceil):
            self.pp1[i] = (self.pp1[i - 1] * self.p1) % self.mod
            self.pp2[i] = (self.pp2[i - 1] * self.p2) % self.mod

        # (x+1, y+1)
        # (i,j) > (i-1, j)p1 (i, j-1)p2 (i-1, j-1) p1p2
        self.left_up = [0] * (self.n + 1) * (self.m + 1)
        for i in range(self.m):
            for j in range(self.n):
                val = self.left_up[i * (self.n + 1) + j + 1] * self.p1 + self.left_up[
                    (i + 1) * (self.n + 1) + j] * self.p2
                val -= self.left_up[i * (self.n + 1) + j] * self.p1 * self.p2 - grid[i * self.n + j]
                self.left_up[(i + 1) * (self.n + 1) + j + 1] = val % self.mod

        # (x+1, y)
        # (i,j) > (i-1, j)p1 (i, j+1)p2 (i-1, j+1) p1p2
        self.right_up = [0] * (self.n + 1) * (self.m + 1)
        for i in range(self.m):
            for j in range(self.n - 1, -1, -1):
                val = self.right_up[i * (self.n + 1) + j] * self.p1 + self.right_up[
                    (i + 1) * (self.n + 1) + j + 1] * self.p2

                val -= self.right_up[i * (self.n + 1) + j + 1] * self.p1 * self.p2 - grid[i * self.n + j]
                self.right_up[(i + 1) * (self.n + 1) + j] = val % self.mod

        # (x, y)
        # (i,j) > (i+1, j)p1 (i, j+1)p2 (i+1, j+1) p1p2
        self.right_down = [0] * (self.n + 1) * (self.m + 1)
        for i in range(self.m - 1, -1, -1):
            for j in range(self.n - 1, -1, -1):
                val = self.right_down[(i + 1) * (self.n + 1) + j] * self.p1 + self.right_down[
                    i * (self.n + 1) + j + 1] * self.p2
                val -= self.right_down[(i + 1) * (self.n + 1) + j + 1] * self.p1 * self.p2 - grid[i * self.n + j]
                self.right_down[i * (self.n + 1) + j] = val % self.mod

        # (x, y+1)
        # (i,j) > (i+1, j)p1 (i, j-1)p2 (i+1, j-1) p1p2
        self.left_down = [0] * (self.n + 1) * (self.m + 1)
        for i in range(self.m - 1, -1, -1):
            for j in range(self.n):
                val = self.left_down[(i + 1) * (self.n + 1) + j + 1] * self.p1 + self.left_down[
                    i * (self.n + 1) + j] * self.p2
                val -= self.left_down[(i + 1) * (self.n + 1) + j] * self.p1 * self.p2 - grid[i * self.n + j]
                self.left_down[i * (self.n + 1) + j + 1] = val % self.mod
        return

    def query_left_up(self, i, j, a, b):
        # (x+1, y+1)
        # (i,j) > (i-a, j)p1 (i, j-b)p2 (i-a, j-b) p1p2
        res = self.left_up[(i + 1) * (self.n + 1) + j + 1]
        res -= self.left_up[(i - a + 1) * (self.n + 1) + j + 1] * self.pp1[a] + self.left_up[
            (i + 1) * (self.n + 1) + j - b + 1] * self.pp2[b]
        res += self.left_up[(i - a + 1) * (self.n + 1) + j - b + 1] * self.pp1[a] * self.pp2[b]
        return res % self.mod

    def query_right_up(self, i, j, a, b):
        # (x+1, y)
        # (i,j) > (i-a, j)p1 (i, j+b)p2 (i-a, j+b) p1p2
        res = self.right_up[(i + 1) * (self.n + 1) + j]
        res -= self.right_up[(i - a + 1) * (self.n + 1) + j] * self.pp1[a] + self.right_up[
            (i + 1) * (self.n + 1) + j + b] * self.pp2[b]
        res += self.right_up[(i - a + 1) * (self.n + 1) + j + b] * self.pp1[a] * self.pp2[b]
        return res % self.mod

    def query_right_down(self, i, j, a, b):
        # (x, y)
        # (i,j) > (i+a, j)p1 (i, j+b)p2 (i+a, j+b) p1p2
        res = self.right_down[i * (self.n + 1) + j]
        res -= self.right_down[(i + a) * (self.n + 1) + j] * self.pp1[a] + self.right_down[i * (self.n + 1) + j + b] * \
               self.pp2[b]
        res += self.right_down[(i + a) * (self.n + 1) + (j + b)] * self.pp1[a] * self.pp2[b]
        return res % self.mod

    def query_left_down(self, i, j, a, b):
        # (x, y+1)
        # (i,j) > (i+a, j)p1 (i, j-b)p2 (i+a, j-b) p1p2
        res = self.left_down[i * (self.n + 1) + j + 1]
        res -= self.left_down[(i + a) * (self.n + 1) + j + 1] * self.pp1[a] + self.left_down[
            i * (self.n + 1) + j - b + 1] * self.pp2[b]
        res += self.left_down[(i + a) * (self.n + 1) + j - b + 1] * self.pp1[a] * self.pp2[b]
        return res % self.mod

    def query_in_build(self, i, j, a, b):
        assert 0 <= i <= i + a - 1 < self.m
        assert 0 <= j <= j + b - 1 < self.n
        res = self.left_up[(i + a) * (self.n + 1) + j + b] - self.left_up[i * (self.n + 1) + j + b] * self.pp1[a] - \
              self.left_up[
                  (i + a) * (self.n + 1) + j] * self.pp2[b]
        res += self.left_up[i * (self.n + 1) + j] * self.pp1[a] * self.pp2[b]
        return res % self.mod


class MatrixHash:
    def __init__(self, m, n, grid):
        """
        primes = PrimeSieve().eratosthenes_sieve(100)
        primes = [x for x in primes if 26 < x < 100]
        """
        primes = [29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        self.m, self.n = m, n

        self.p1 = primes[random.randint(0, len(primes) - 1)]
        while True:
            self.p2 = primes[random.randint(0, len(primes) - 1)]
            if self.p2 != self.p1:
                break

        ceil = self.m if self.m > self.n else self.n
        self.pp1 = [1] * (ceil + 1)
        self.pp2 = [1] * (ceil + 1)
        self.mod = random.randint(10 ** 9 + 7, (1 << 31) - 1)

        for i in range(1, ceil):
            self.pp1[i] = (self.pp1[i - 1] * self.p1) % self.mod
            self.pp2[i] = (self.pp2[i - 1] * self.p2) % self.mod

        self.pre = [0] * (self.n + 1) * (self.m + 1)

        for i in range(self.m):
            for j in range(self.n):
                val = self.pre[i * (self.n + 1) + j + 1] * self.p1 + self.pre[(i + 1) * (self.n + 1) + j] * self.p2
                val -= self.pre[i * (self.n + 1) + j] * self.p1 * self.p2 - grid[i * self.n + j]
                self.pre[(i + 1) * (self.n + 1) + j + 1] = val % self.mod
        return

    def query_sub(self, i, j, a, b):
        # right_down corner
        assert a - 1 <= i < self.m
        assert b - 1 <= j < self.n
        res = self.pre[(i + 1) * (self.n + 1) + j + 1]
        res -= self.pre[(i - a + 1) * (self.n + 1) + j + 1] * self.pp1[a] + self.pre[
            (i + 1) * (self.n + 1) + j - b + 1] * self.pp2[b]
        res += self.pre[(i - a + 1) * (self.n + 1) + j - b + 1] * self.pp1[a] * self.pp2[b]
        return res % self.mod

    def query_matrix(self, a, b, mat):
        cur = [0] * (b + 1) * (a + 1)
        for i in range(a):
            for j in range(b):
                val = cur[i * (b + 1) + j + 1] * self.p1 + cur[(i + 1) * (b + 1) + j] * self.p2
                val -= cur[i * (b + 1) + j] * self.p1 * self.p2 - mat[i * b + j]
                cur[(i + 1) * (b + 1) + j + 1] = val % self.mod
        return cur[-1]


class StringHash:
    def __init__(self, lst):
        """two mod to avoid hash crush"""
        # use two class to compute is faster!!!
        self.n = len(lst)
        self.p = random.randint(26, 100)
        self.mod = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        self.pre = [0] * (self.n + 1)
        self.pp = [1] * (self.n + 1)
        for j, w in enumerate(lst):
            self.pre[j + 1] = (self.pre[j] * self.p + w) % self.mod
            self.pp[j + 1] = (self.pp[j] * self.p) % self.mod
        return

    def query(self, x, y):
        """range hash value index start from 0"""
        # assert 0 <= x <= y <= self.n - 1
        if y < x:
            return 0
        # with length y - x + 1 important!!!
        ans = (self.pre[y + 1] - self.pre[x] * self.pp[y - x + 1]) % self.mod
        return ans, y - x + 1


class StringHashSingle:
    def __init__(self, lst):
        """two mod to avoid hash crush"""
        # use two class to compute is faster!!!
        self.n = len(lst)
        base = max(lst) + 1
        self.p = random.randint(base, base * 2)
        self.mod = random.getrandbits(64)
        self.mod = random.getrandbits(64)

        self.pre = [0] * (self.n + 1)
        self.pp = [1] * (self.n + 1)
        for j, w in enumerate(lst):
            self.pre[j + 1] = (self.pre[j] * self.p + w) % self.mod
            self.pp[j + 1] = (self.pp[j] * self.p) % self.mod
        return

    def query(self, x, y):
        """range hash value index start from 0"""
        # assert 0 <= x <= y <= self.n - 1
        if y < x:
            return 0
        # with length y - x + 1 important!!!
        ans = (self.pre[y + 1] - self.pre[x] * self.pp[y - x + 1]) % self.mod
        return ans, y - x + 1


class PointSetRangeHashReverse:
    def __init__(self, n) -> None:
        self.n = n
        self.p = random.randint(26, 100)
        self.mod = random.randint(10 ** 9 + 7, (1 << 31) - 1)
        self.pp = [1] * (n + 1)
        for j in range(n):
            self.pp[j + 1] = (self.pp[j] * self.p) % self.mod
        self.left_to_right = [0] * (4 * n)
        self.right_to_left = [0] * (4 * n)
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.left_to_right[i] = nums[s]
                    self.right_to_left[i] = nums[s]
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i, s, t)
        return

    def _push_up(self, i: int, s, t) -> None:
        m = s + (t - s) // 2
        length = t - m
        self.left_to_right[i] = (self.left_to_right[i << 1] * self.pp[length] + self.left_to_right[
            (i << 1) | 1]) % self.mod

        length = m - s + 1
        self.right_to_left[i] = (self.right_to_left[(i << 1) | 1] * self.pp[length] + self.right_to_left[
            i << 1]) % self.mod
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.left_to_right[i]
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.right_to_left[i] = self.left_to_right[i] = val
                    continue
                m = s + (t - s) // 2
                stack.append((s, t, ~i))
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i, s, t)
        return

    def range_hash(self, left: int, right: int):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                length = t - s + 1
                ans = (ans * self.pp[length] + self.left_to_right[i]) % self.mod
                continue
            m = s + (t - s) // 2
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans

    def range_hash_reverse(self, left: int, right: int):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                length = t - s + 1
                ans = (ans * self.pp[length] + self.right_to_left[i]) % self.mod
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class RangeSetRangeHashReverse:
    def __init__(self, n, tag=inf) -> None:
        self.n = n
        self.tag = tag
        while True:
            self.p = random.randint(26, 100)
            self.mod = random.randint(10 ** 9 + 7, (1 << 31) - 1)
            if math.gcd(self.p - 1, self.mod) == 1:
                break
        self.pp = [1] * (n + 1)
        for j in range(n):
            self.pp[j + 1] = (self.pp[j] * self.p) % self.mod
        self.rev = pow(self.p - 1, -1, self.mod)
        self.left_to_right = [0] * (4 * n)
        self.right_to_left = [0] * (4 * n)
        self.lazy = [self.tag] * (4 * self.n)
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(nums[s], i, s, t)
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i, s, t)
        return

    def _make_tag(self, val: int, i: int, s, t) -> None:
        self.lazy[i] = val
        m = t - s + 1
        self.left_to_right[i] = (val * (self.pp[m] - 1) * self.rev) % self.mod
        self.right_to_left[i] = (val * (self.pp[m] - 1) * self.rev) % self.mod
        return

    def _push_down(self, i: int, s, t) -> None:
        m = s + (t - s) // 2
        if self.lazy[i] != self.tag:
            self._make_tag(self.lazy[i], i << 1, s, m)
            self._make_tag(self.lazy[i], (i << 1) | 1, m + 1, t)
            self.lazy[i] = self.tag
        return

    def _push_up(self, i: int, s, t) -> None:
        m = s + (t - s) // 2
        length = t - m
        self.left_to_right[i] = (self.left_to_right[i << 1] * self.pp[length] + self.left_to_right[
            (i << 1) | 1]) % self.mod

        length = m - s + 1
        self.right_to_left[i] = (self.right_to_left[(i << 1) | 1] * self.pp[length] + self.right_to_left[
            (i << 1)]) % self.mod
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.left_to_right[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, t)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_set(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(val, i, s, t)
                    continue
                m = s + (t - s) // 2
                self._push_down(i, s, t)
                stack.append((s, t, ~i))

                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i, s, t)
        return

    def range_hash(self, left: int, right: int):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                length = t - s + 1
                ans = (ans * self.pp[length] + self.left_to_right[i]) % self.mod
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, t)
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans

    def range_hash_reverse(self, left: int, right: int):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                length = t - s + 1
                ans = (ans * self.pp[length] + self.right_to_left[i]) % self.mod
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans
