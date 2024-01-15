import math
import random

from src.utils.fast_io import inf


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
        return ans


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


class RangeChangeRangeHashReverse:
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

    def range_change(self, left: int, right: int, val: int) -> None:
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
