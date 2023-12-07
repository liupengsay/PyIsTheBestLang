from collections import defaultdict
from math import inf
from typing import List


class SegTreeBrackets:
    def __init__(self, n, s):
        self.n = n
        self.s = s
        self.a = [0] * (2 * self.n)
        self.b = [0] * (2 * self.n)
        self.c = [0] * (2 * self.n)

    def build(self):
        for i in range(self.n):
            self.a[i + self.n] = 0
            self.b[i + self.n] = 1 if self.s[i] == "(" else 0
            self.c[i + self.n] = 1 if self.s[i] == ")" else 0
        for i in range(self.n - 1, 0, -1):
            t = min(self.b[i << 1], self.c[i << 1 | 1])
            self.a[i] = self.a[i << 1] + self.a[i << 1 | 1] + 2 * t
            self.b[i] = self.b[i << 1] + self.b[i << 1 | 1] - t
            self.c[i] = self.c[i << 1] + self.c[i << 1 | 1] - t

    def query(self, low, r):
        left = []
        right = []
        low += self.n
        r += self.n
        while low <= r:
            if low & 1:
                left.append([self.a[low], self.b[low], self.c[low]])
                low += 1
            if not r & 1:
                right.append([self.a[r], self.b[r], self.c[r]])
                r -= 1
            low >>= 1
            r >>= 1
        a1 = b1 = c1 = 0
        for a2, b2, c2 in left + right[::-1]:
            t = b1 if b1 < c2 else c2
            a1 += a2 + 2 * t
            b1 += b2 - t
            c1 += c2 - t
        return a1


class RangeAscendRangeMax:
    def __init__(self, n):
        self.n = n
        self.cover = [-inf] * (4 * n)
        self.lazy = [-inf] * (4 * n)

    @staticmethod
    def _max(a, b):
        return a if a > b else b

    def _make_tag(self, i, val) -> None:
        self.cover[i] = self._max(self.cover[i], val)
        self.lazy[i] = self._max(self.lazy[i], val)
        return

    def _push_up(self, i):
        self.cover[i] = self._max(self.cover[2 * i], self.cover[2 * i + 1])
        return

    def _push_down(self, i):
        if self.lazy[i] != -inf:
            self.cover[2 * i] = self._max(self.cover[2 * i], self.lazy[i])
            self.cover[2 * i + 1] = self._max(self.cover[2 * i + 1], self.lazy[i])
            self.lazy[2 * i] = self._max(self.lazy[2 * i], self.lazy[i])
            self.lazy[2 * i + 1] = self._max(self.lazy[2 * i + 1], self.lazy[i])
            self.lazy[i] = -inf
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self._make_tag(ind, nums[s])
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self._push_up(ind)
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, 2 * i))
            stack.append((m + 1, t, 2 * i + 1))
        return nums

    def range_ascend(self, left, right, val):
        # update the range ascend
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            a, b, i = stack.pop()
            if i >= 0:
                if left <= a and b <= right:
                    self._make_tag(i, val)
                    continue
                self._push_down(i)
                stack.append([a, b, ~i])
                m = a + (b - a) // 2
                if left <= m:
                    stack.append([a, m, 2 * i])
                if right > m:
                    stack.append([m + 1, b, 2 * i + 1])
            else:
                i = ~i
                self._push_up(i)
        return

    def range_max(self, left, right):
        # query the range max
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        highest = -inf
        while stack:
            a, b, i = stack.pop()
            if left <= a and b <= right:
                highest = self._max(highest, self.cover[i])
                continue
            self._push_down(i)
            m = a + (b - a) // 2
            if left <= m:
                stack.append([a, m, 2 * i])
            if right > m:
                stack.append([m + 1, b, 2 * i + 1])
        return highest


class RangeAscendRangeMaxBinarySearchFindLeft:
    def __init__(self, n):
        self.n = n
        self.cover = [-inf] * (4 * n)
        self.lazy = [-inf] * (4 * n)

    @staticmethod
    def _max(a, b):
        return a if a > b else b

    def _make_tag(self, i, val) -> None:
        self.cover[i] = self._max(self.cover[i], val)
        self.lazy[i] = self._max(self.lazy[i], val)
        return

    def _push_up(self, i):
        self.cover[i] = self._max(self.cover[2 * i], self.cover[2 * i + 1])
        return

    def _push_down(self, i):
        if self.lazy[i] != -inf:
            self.cover[2 * i] = self._max(self.cover[2 * i], self.lazy[i])
            self.cover[2 * i + 1] = self._max(self.cover[2 * i + 1], self.lazy[i])
            self.lazy[2 * i] = self._max(self.lazy[2 * i], self.lazy[i])
            self.lazy[2 * i + 1] = self._max(self.lazy[2 * i + 1], self.lazy[i])
            self.lazy[i] = -inf
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self._make_tag(ind, nums[s])
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self._push_up(ind)
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, 2 * i))
            stack.append((m + 1, t, 2 * i + 1))
        return nums

    def range_ascend(self, left, right, val):
        # update the range ascend
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            a, b, i = stack.pop()
            if i >= 0:
                if left <= a and b <= right:
                    self._make_tag(i, val)
                    continue
                self._push_down(i)
                stack.append([a, b, ~i])
                m = a + (b - a) // 2
                if left <= m:
                    stack.append([a, m, 2 * i])
                if right > m:
                    stack.append([m + 1, b, 2 * i + 1])
            else:
                i = ~i
                self._push_up(i)
        return

    def range_max(self, left, right):
        # query the range max
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        highest = -inf
        while stack:
            a, b, i = stack.pop()
            if left <= a and b <= right:
                highest = self._max(highest, self.cover[i])
                continue
            self._push_down(i)
            m = a + (b - a) // 2
            if left <= m:
                stack.append([a, m, 2 * i])
            if right > m:
                stack.append([m + 1, b, 2 * i + 1])
        return highest

    def binary_search_find_left(self, left, right, val):
        """binary search with segment tree like"""
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        res = -1
        while stack and res == -1:
            a, b, i = stack.pop()
            if left <= a == b <= right:
                if self.cover[i] >= val:
                    res = a
                continue
            self._push_down(i)
            m = a + (b - a) // 2
            if right > m and self.cover[2 * i + 1] >= val:
                stack.append([m + 1, b, 2 * i + 1])
            if left <= m and self.cover[2 * i] >= val:
                stack.append([a, m, 2 * i])
        return res


class RangeDescendRangeMin:
    def __init__(self, n):
        self.n = n
        self.cover = [inf] * (4 * n)
        self.lazy = [inf] * (4 * n)

    @staticmethod
    def _min(a, b):
        return a if a < b else b

    def _make_tag(self, i, val) -> None:
        self.cover[i] = self._min(self.cover[i], val)
        self.lazy[i] = self._min(self.lazy[i], val)
        return

    def _push_up(self, i):
        self.cover[i] = self._min(self.cover[2 * i], self.cover[2 * i + 1])
        return

    def _push_down(self, i):
        if self.lazy[i] != inf:
            self.cover[2 * i] = self._min(self.cover[2 * i], self.lazy[i])
            self.cover[2 * i + 1] = self._min(self.cover[2 * i + 1], self.lazy[i])
            self.lazy[2 * i] = self._min(self.lazy[2 * i], self.lazy[i])
            self.lazy[2 * i + 1] = self._min(self.lazy[2 * i + 1], self.lazy[i])
            self.lazy[i] = inf
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self._make_tag(ind, nums[s])
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self._push_up(ind)
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, 2 * i))
            stack.append((m + 1, t, 2 * i + 1))
        return nums

    def range_descend(self, left, right, val):
        # update the range descend
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            a, b, i = stack.pop()
            if i >= 0:
                if left <= a and b <= right:
                    self._make_tag(i, val)
                    continue
                self._push_down(i)
                stack.append([a, b, ~i])
                m = a + (b - a) // 2
                if left <= m:
                    stack.append([a, m, 2 * i])
                if right > m:
                    stack.append([m + 1, b, 2 * i + 1])
            else:
                i = ~i
                self._push_up(i)
        return

    def range_min(self, left, right):
        # query the range min
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        lowest = inf
        while stack:
            a, b, i = stack.pop()
            if left <= a and b <= right:
                lowest = self._min(lowest, self.cover[i])
                continue
            self._push_down(i)
            m = a + (b - a) // 2
            if left <= m:
                stack.append([a, m, 2 * i])
            if right > m:
                stack.append([m + 1, b, 2 * i + 1])
        return lowest


class RangeAddRangeSumMinMax:
    def __init__(self, n) -> None:
        self.n = n
        self.cover = [0] * (4 * self.n)  # range sum
        self.lazy = [0] * (4 * self.n)  # lazy tag
        self.floor = [0] * (4 * self.n)  # range min
        self.ceil = [0] * (4 * self.n)  # range max
        return

    @staticmethod
    def _max(a: int, b: int) -> int:
        return a if a > b else b

    @staticmethod
    def _min(a: int, b: int) -> int:
        return a if a < b else b

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self._make_tag(ind, s, t, nums[s])
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self._push_up(ind)
        return

    def _push_down(self, i: int, s: int, m: int, t: int) -> None:
        if self.lazy[i]:
            self.cover[2 * i] += self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] += self.lazy[i] * (t - m)

            self.floor[2 * i] += self.lazy[i]
            self.floor[2 * i + 1] += self.lazy[i]

            self.ceil[2 * i] += self.lazy[i]
            self.ceil[2 * i + 1] += self.lazy[i]

            self.lazy[2 * i] += self.lazy[i]
            self.lazy[2 * i + 1] += self.lazy[i]

            self.lazy[i] = 0

    def _push_up(self, i) -> None:
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.ceil[i] = self._max(self.ceil[2 * i], self.ceil[2 * i + 1])
        self.floor[i] = self._min(self.floor[2 * i], self.floor[2 * i + 1])
        return

    def _make_tag(self, i, s, t, val) -> None:
        self.cover[i] += val * (t - s + 1)
        self.floor[i] += val
        self.ceil[i] += val
        self.lazy[i] += val
        return

    def range_add(self, left: int, right: int, val: int) -> None:
        # update the range add
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, s, t, val)
                    continue

                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                stack.append((s, t, ~i))

                if left <= m:
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_sum(self, left: int, right: int) -> int:
        # query the range sum
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans

    def range_min(self, left: int, right: int) -> int:
        # query the range min
        stack = [(0, self.n - 1, 1)]
        lowest = inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                lowest = self._min(lowest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return lowest

    def range_max(self, left: int, right: int) -> int:
        # query the rang max
        stack = [(0, self.n - 1, 1)]
        highest = -inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self._max(highest, self.ceil[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return highest

    def get(self) -> List[int]:
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            stack.append((s, m, 2 * i))
            stack.append((m + 1, t, 2 * i + 1))
        return nums


class RangeChangeRangeSumMinMax:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * self.n)  # range sum
        self.lazy = [inf] * (4 * self.n)  # because range change can to be 0 the lazy tag must be inf
        self.floor = [inf] * (4 * self.n)  # because range change can to be any integer the floor initial must be inf
        self.ceil = [-inf] * (4 * self.n)  # because range change can to be any integer the ceil initial must be -inf
        return

    @staticmethod
    def _max(a, b):
        return a if a > b else b

    @staticmethod
    def _min(a, b):
        return a if a < b else b

    def _push_down(self, i, s, m, t):
        if self.lazy[i] != inf:
            self.cover[2 * i] = self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] = self.lazy[i] * (t - m)

            self.floor[2 * i] = self.lazy[i]
            self.floor[2 * i + 1] = self.lazy[i]

            self.ceil[2 * i] = self.lazy[i]
            self.ceil[2 * i + 1] = self.lazy[i]

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = inf

    def _push_up(self, i) -> None:
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.ceil[i] = self._max(self.ceil[2 * i], self.ceil[2 * i + 1])
        self.floor[i] = self._min(self.floor[2 * i], self.floor[2 * i + 1])
        return

    def _make_tag(self, i, s, t, val) -> None:
        self.cover[i] = val * (t - s + 1)
        self.floor[i] = val
        self.ceil[i] = val
        self.lazy[i] = val
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self._make_tag(ind, s, t, nums[s])
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self._push_up(ind)
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            stack.append((s, m, 2 * i))
            stack.append((m + 1, t, 2 * i + 1))
        return nums

    def range_change(self, left, right, val):
        # update the range change
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, s, t, val)
                    continue

                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                stack.append((s, t, ~i))

                if left <= m:
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_sum(self, left, right):
        # query the range sum
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans

    def range_min(self, left, right):
        # query the range min
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        lowest = inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                lowest = self._min(lowest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return lowest

    def range_max(self, left, right):
        # query the range max
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        highest = -inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self._max(highest, self.ceil[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return highest


class RangeChangeRangeSumMinMaxDynamic:
    def __init__(self, n):
        # dynamic adding point segment tree in which n can be 1e9
        self.n = n
        self.cover = defaultdict(int)  # range sum must be initial 0
        self.lazy = defaultdict(lambda: inf)  # lazy tag must be initial inf
        self.floor = defaultdict(int)  # range min can be inf
        self.ceil = defaultdict(int)  # range max can be -inf
        return

    @staticmethod
    def _max(a, b):
        return a if a > b else b

    @staticmethod
    def _min(a, b):
        return a if a < b else b

    def _push_down(self, i, s, m, t):
        if self.lazy[i] != inf:
            self.cover[2 * i] = self.lazy[i] * (m - s + 1)
            self.cover[2 * i + 1] = self.lazy[i] * (t - m)

            self.floor[2 * i] = self.lazy[i]
            self.floor[2 * i + 1] = self.lazy[i]

            self.ceil[2 * i] = self.lazy[i]
            self.ceil[2 * i + 1] = self.lazy[i]

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = inf

    def _push_up(self, i) -> None:
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.ceil[i] = self._max(self.ceil[2 * i], self.ceil[2 * i + 1])
        self.floor[i] = self._min(self.floor[2 * i], self.floor[2 * i + 1])
        return

    def _make_tag(self, i, s, t, val) -> None:
        self.cover[i] = val * (t - s + 1)
        self.floor[i] = val
        self.ceil[i] = val
        self.lazy[i] = val
        return

    def range_change(self, left, right, val):
        # update the range change
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, s, t, val)
                    continue

                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                stack.append((s, t, ~i))

                if left <= m:
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_sum(self, left, right):
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans

    def range_min(self, left, right):
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        highest = inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self._min(highest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return highest

    def range_max(self, left, right):
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        highest = -inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self._max(highest, self.ceil[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return highest


class SegmentTreeRangeUpdateChangeQueryMax:
    def __init__(self, nums: List[int]) -> None:
        # 区间值增减、区间值修改、区间最大值查询
        self.n = len(nums)
        self.nums = nums
        self.lazy = [[inf, 0]] * (4 * self.n)  # 懒标记
        self.ceil = [-inf] * (4 * self.n)  # 最大值
        self.build()  # 初始化线段树
        return

    @staticmethod
    def _max(a: int, b: int) -> int:
        return a if a > b else b

    @staticmethod
    def _min(a: int, b: int) -> int:
        return a if a < b else b

    def build(self) -> None:
        # 数组初始化线段树
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.ceil[ind] = self.nums[s]
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self.ceil[ind] = self._max(self.ceil[2 * ind], self.ceil[2 * ind + 1])
        return

    def _push_down(self, i: int, s: int, m: int, t: int) -> None:
        # 下放懒标记
        if self.lazy[i] != [inf, 0]:
            a, b = self.lazy[i]  # 分别表示修改为 a 与 增| b
            if a == inf:
                self.ceil[2 * i] += b
                self.ceil[2 * i + 1] += b
                self.lazy[2 * i] = [inf, self.lazy[2 * i][1] + b]
                self.lazy[2 * i + 1] = [inf, self.lazy[2 * i + 1][1] + b]
            else:
                self.ceil[2 * i] = a
                self.ceil[2 * i + 1] = a
                self.lazy[2 * i] = [a, 0]
                self.lazy[2 * i + 1] = [a, 0]
            self.lazy[i] = [inf, 0]

    def update(self, left: int, right: int, s: int, t: int, val: int, flag: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    if flag == 1:
                        self.ceil[i] = val
                        self.lazy[i] = [val, 0]
                    elif self.lazy[i][0] != inf:
                        self.ceil[i] += val
                        self.lazy[i] = [self.lazy[i][0] + val, 0]
                    else:
                        self.ceil[i] += val
                        self.lazy[i] = [inf, self.lazy[i][1] + val]
                    continue

                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                stack.append((s, t, ~i))

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self.ceil[i] = self._max(self.ceil[2 * i], self.ceil[2 * i + 1])
        return

    def query_max(self, left: int, right: int, s: int, t: int, i: int) -> int:

        # 查询区间的最大值
        stack = [[s, t, i]]
        highest = -inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = self._max(highest, self.ceil[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return highest


class RangeKSmallest:
    def __init__(self, n, k) -> None:
        """query the k smallest value of static range which can also change to support dynamic"""
        self.n = n
        self.k = k
        self.cover = [[] for _ in range(4 * self.n)]
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.cover[ind].append(nums[s])
                    continue
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self._push_up(ind)
        return

    def _merge(self, lst1, lst2):
        res = []
        m, n = len(lst1), len(lst2)
        i = j = 0
        while i < m and j < n:
            if lst1[i] < lst2[j]:
                res.append(lst1[i])
                i += 1
            else:
                res.append(lst2[j])
                j += 1
        res.extend(lst1[i:])
        res.extend(lst2[j:])
        return res[:self.k]

    def _push_up(self, i) -> None:
        self.cover[i] = self._merge(self.cover[2 * i][:], self.cover[2 * i + 1][:])
        return

    def range_k_smallest(self, left: int, right: int) -> int:
        # query the range sum
        stack = [(0, self.n - 1, 1)]
        ans = []
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = self._merge(ans, self.cover[i][:])
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans


class RangeOrRangeAnd:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * 4 * n
        self.lazy = [0] * 4 * n
        return

    def _make_tag(self, i, val) -> None:
        self.cover[i] |= val
        self.lazy[i] |= val
        return

    def _push_down(self, i):
        if self.lazy[i]:
            self.cover[2 * i] |= self.lazy[i]
            self.cover[2 * i + 1] |= self.lazy[i]

            self.lazy[2 * i] |= self.lazy[i]
            self.lazy[2 * i + 1] |= self.lazy[i]

            self.lazy[i] = 0

    def _push_up(self, i):
        self.cover[i] = self.cover[2 * i] & self.cover[2 * i + 1]
        return

    def build(self, nums: List[int]) -> None:
        # 数组初始化线段树
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self._make_tag(ind, nums[s])
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self._push_up(ind)
        return

    def range_or(self, left, r, val):
        """update the range or"""
        assert 0 <= left <= r <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= r:
                    self.cover[i] |= val
                    self.lazy[i] |= val
                    continue
                m = s + (t - s) // 2
                self._push_down(i)
                stack.append((s, t, ~i))
                if left <= m:
                    stack.append((s, m, 2 * i))
                if r > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_and(self, left, r):
        """query the range and"""
        assert 0 <= left <= r <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        ans = (1 << 31) - 1
        while stack and ans:
            s, t, i = stack.pop()
            if left <= s and t <= r:
                ans &= self.cover[i]
                continue
            self._push_down(i)
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, 2 * i))
            if r > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, 2 * i))
            stack.append((m + 1, t, 2 * i + 1))
        return nums


class SegmentTreeRangeUpdateXORSum:
    def __init__(self, n):
        # 区间值01翻转与区间和查询
        self.n = n
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [0] * (4 * self.n)  # 懒标记
        return

    def build(self, nums) -> None:
        # 数组初始化线段树
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, 2 * i))
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def _push_down(self, i: int, s: int, m: int, t: int) -> None:
        if self.lazy[i]:
            self.cover[2 * i] = m - s + 1 - self.cover[2 * i]
            self.cover[2 * i + 1] = t - m - self.cover[2 * i + 1]

            self.lazy[2 * i] ^= self.lazy[i]  # 注意异或抵消查询
            self.lazy[2 * i + 1] ^= self.lazy[i]  # 注意异或抵消查询

            self.lazy[i] = 0
        return

    def update_range(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = t - s + 1 - self.cover[i]
                    self.lazy[i] ^= val  # 注意异或抵消查询
                    continue

                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                stack.append((s, t, ~i))

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query_sum(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans


class SegmentTreeRangeAddMulSum:

    def __init__(self, p, n):
        self.p = p
        # 区间值增减乘积与区间和查询
        self.cover = [0] * 4 * n
        self.lazy = [[] for _ in range(4 * n)]

    def _push_down(self, i, s, m, t):

        if self.lazy[i]:
            for op, val in self.lazy[i]:
                if op == "add":
                    self.cover[2 * i] += val * (m - s + 1)
                    self.cover[2 * i + 1] += val * (t - m)

                    self.lazy[2 * i] += [[op, val]]
                    self.lazy[2 * i + 1] += [[op, val]]
                else:
                    self.cover[2 * i] *= val
                    self.cover[2 * i + 1] *= val

                    self.lazy[2 * i] += [[op, val]]
                    self.lazy[2 * i + 1] += [[op, val]]
                self.cover[2 * i] %= self.p
                self.cover[2 * i + 1] %= self.p

            self.lazy[i] = []

    def update(self, left, r, s, t, op, val, i):
        if left <= s and t <= r:
            if op == "add":
                self.cover[i] += val * (t - s + 1)
                self.lazy[i] += [["add", val]]
            else:
                self.cover[i] *= val
                self.lazy[i] += [["mul", val]]
            self.cover[i] %= self.p
            return
        m = s + (t - s) // 2
        self._push_down(i, s, m, t)
        if left <= m:
            self.update(left, r, s, m, op, val, 2 * i)
        if r > m:
            self.update(left, r, m + 1, t, op, val, 2 * i + 1)
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.cover[i] %= self.p
        return

    def query(self, left, r, s, t, i):
        if left <= s and t <= r:
            return self.cover[i]
        m = s + (t - s) // 2
        self._push_down(i, s, m, t)
        ans = 0
        if left <= m:
            ans += self.query(left, r, s, m, 2 * i)
        if r > m:
            ans += self.query(left, r, m + 1, t, 2 * i + 1)
        return ans


class RangeChangeRangeOr:
    def __init__(self, n) -> None:
        # 区间值修改，区间或查询
        self.n = n
        self.lazy = [inf] * (4 * self.n)  # 懒标记
        self.cover = [0] * (4 * self.n)  # 区间或
        return

    def _make_tag(self, val: int, i: int) -> None:
        self.cover[i] = val
        self.lazy[i] = val
        return

    def _push_down(self, i: int) -> None:
        # 下放懒标记
        if self.lazy[i] != inf:
            self._make_tag(self.lazy[i], 2 * i)
            self._make_tag(self.lazy[i], 2 * i + 1)
            self.lazy[i] = inf

    def _push_up(self, i: int) -> None:
        self.cover[i] = self.cover[2 * i] | self.cover[2 * i + 1]

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self._make_tag(nums[s], ind)
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self._push_up(ind)
        return

    def get(self):
        # 查询区间的所有值
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, 2 * i))
            stack.append((m + 1, t, 2 * i + 1))
        return nums

    def range_change(self, left: int, right: int, val: int) -> None:
        # 修改区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(val, i)
                    continue
                m = s + (t - s) // 2
                self._push_down(i)
                stack.append((s, t, ~i))

                if left <= m:
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_or(self, left: int, right: int) -> int:
        # 查询区间的和
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans |= self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans


class SegmentTreeRangeUpdateMulQuerySum:
    def __init__(self, nums: List[int], p) -> None:
        # 区间值增减、区间值乘法、区间值修改、区间最大值查询
        self.p = p
        self.n = len(nums)
        self.nums = nums
        self.lazy_add = [0] * (4 * self.n)  # 懒标记
        self.lazy_mul = [1] * (4 * self.n)  # 懒标记
        self.cover = [0] * (4 * self.n)  # 区间和
        self.build()  # 初始化线段树
        return

    @staticmethod
    def _max(a: int, b: int) -> int:
        return a if a > b else b

    @staticmethod
    def _min(a: int, b: int) -> int:
        return a if a < b else b

    def build(self) -> None:
        # 数组初始化线段树
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = self.nums[s]
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, 2 * i))
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def _make_tag(self, s: int, t: int, x: int, flag: int, i: int) -> None:
        if flag == 1:  # 乘法
            self.lazy_mul[i] = (self.lazy_mul[i] * x) % self.p
            self.lazy_add[i] = (self.lazy_add[i] * x) % self.p
            self.cover[i] = (self.cover[i] * x) % self.p
        else:
            self.lazy_add[i] = (self.lazy_add[i] + x) % self.p
            self.cover[i] = (self.cover[i] + x * (t - s + 1)) % self.p
        return

    def _push_down(self, i: int, s: int, m: int, t: int) -> None:
        # 下放懒标记
        if self.lazy_mul[i] != 1:
            self._make_tag(s, m, self.lazy_mul[i], 1, 2 * i)
            self._make_tag(m + 1, t, self.lazy_mul[i], 1, 2 * i + 1)
            self.lazy_mul[i] = 1

        if self.lazy_add[i] != 0:
            self._make_tag(s, m, self.lazy_add[i], 2, 2 * i)
            self._make_tag(m + 1, t, self.lazy_add[i], 2, 2 * i + 1)
            self.lazy_add[i] = 0

    def update(self, left: int, right: int, s: int, t: int, val: int, flag: int, i: int) -> None:
        # 增减区间值 left 与 right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    if flag == 1:
                        self.cover[i] = (self.cover[i] * val) % self.p
                        self.lazy_add[i] = (self.lazy_add[i] * val) % self.p
                        self.lazy_mul[i] = (self.lazy_mul[i] * val) % self.p
                    else:
                        self.cover[i] = (self.cover[i] + val * (t - s + 1)) % self.p
                        self.lazy_add[i] = (self.lazy_add[i] + val) % self.p
                    continue

                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                stack.append((s, t, ~i))

                if left <= m:  # 注意左右子树的边界与范围
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
                self.cover[i] %= self.p
        return

    def query_sum(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans % self.p


class SegmentTreePointUpdateRangeMulQuery:
    def __init__(self, n, mod) -> None:
        # 单点值修改、区间乘mod|
        self.n = n
        self.mod = mod
        self.cover = [1] * (4 * self.n)  # 区间乘积mod|
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改单点值 left == right 取值为 0 到 n-1 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = val
                    continue
                m = s + (t - s) // 2
                stack.append((s, t, ~i))
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self.cover[i] = self.cover[2 * i] * self.cover[2 * i + 1]
                self.cover[i] %= self.mod
        return

    def query_mul(self, left: int, right: int, s: int, t: int, i: int) -> int:
        # 查询区间的乘积
        stack = [[s, t, i]]
        ans = 1
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans *= self.cover[i]
                ans %= self.mod
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans


class PointChangeRangeMaxNonEmpConSubSum:
    def __init__(self, n, initial=inf) -> None:
        # 区间值修改、区间最大非空连续子段和查询
        self.n = n
        self.initial = initial
        self.cover = [-initial] * (4 * self.n)
        self.left = [-initial] * (4 * self.n)
        self.right = [-initial] * (4 * self.n)
        self.lazy = [initial] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        return

    @staticmethod
    def _max(a, b):
        return a if a > b else b

    def _make_tag(self, i, val):
        self.cover[i] = val
        self.left[i] = val
        self.right[i] = val
        self.sum[i] = val
        self.lazy[i] = val
        return

    def _push_down(self, i):
        if self.lazy[i] != self.initial:
            self._make_tag(2 * i, self.lazy[i])
            self._make_tag(2 * i + 1, self.lazy[i])
            self.lazy[i] = self.initial
        return

    def _merge(self, res1, res2):
        res = [0] * 4
        res[0] = self._max(res1[0], res2[0])
        res[0] = self._max(res[0], res1[2] + res2[1])
        res[1] = self._max(res1[1], res1[3] + res2[1])
        res[2] = self._max(res2[2], res2[3] + res1[2])
        res[3] = res1[3] + res2[3]
        return res

    def _push_up(self, i):
        res1 = self.cover[2 * i], self.left[2 * i], self.right[2 * i], self.sum[2 * i]
        res2 = self.cover[2 * i + 1], self.left[2 * i + 1], self.right[2 * i + 1], self.sum[2 * i + 1]
        self.cover[i], self.left[i], self.right[i], self.sum[i] = self._merge(res1, res2)
        return

    def build(self, nums: List[int]) -> None:
        assert len(nums) == self.n
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, nums[s])
                    continue
                stack.append([s, t, ~i])
                m = s + (t - s) // 2
                stack.append([s, m, 2 * i])
                stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self._push_up(i)
        return

    def get(self) -> List[int]:
        stack = [(0, self.n - 1, 1)]
        nums = [-1] * self.n
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    nums[s] = self.cover[i]
                    continue
                stack.append([s, t, ~i])
                m = s + (t - s) // 2
                self._push_down(i)
                stack.append([s, m, 2 * i])
                stack.append([m + 1, t, 2 * i + 1])
            else:
                i = ~i
                self._push_up(i)
        return nums

    def range_change(self, left: int, right: int, val: int) -> None:
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, val)
                    continue
                m = s + (t - s) // 2
                stack.append((s, t, ~i))
                self._push_down(i)
                if left <= m:
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_max_non_emp_con_sub_sum(self, left: int, right: int) -> List[int]:
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        dct = defaultdict(lambda: [-self.initial] * 4)
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    dct[i] = [self.cover[i], self.left[i], self.right[i], self.sum[i]]
                    continue
                stack.append((s, t, ~i))
                self._push_down(i)
                m = s + (t - s) // 2
                if left <= m:
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                dct[i] = self._merge(dct[2 * i], dct[2 * i + 1])
        return dct[1]


class SegmentTreeRangeUpdateAvgDev:
    def __init__(self, n) -> None:
        # 区间增减、区间平均值与区间方差
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.cover_2 = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        return

    @staticmethod
    def _max(a, b):
        return a if a > b else b

    @staticmethod
    def _min(a: int, b: int) -> int:
        return a if a < b else b

    def _push_up(self, i):
        # 合并区间的函数
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        self.cover_2[i] = self.cover_2[2 * i] + self.cover_2[2 * i + 1]
        return

    def _make_tag(self, s, t, i, val):
        self.cover_2[i] += self.cover[i] * 2 * val + (t - s + 1) * val * val
        self.cover[i] += val * (t - s + 1)
        self.lazy[i] += val
        return

    def _push_down(self, i, s, m, t):
        if self.lazy[i]:
            self._make_tag(s, m, 2 * i, self.lazy[i])
            self._make_tag(m + 1, t, 2 * i + 1, self.lazy[i])
            self.lazy[i] = 0

    def build(self, nums) -> None:
        # 数组初始化线段树
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                    self.cover_2[i] = nums[s] * nums[s]
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                stack.append((s, m, 2 * i))
                stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 增| val 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(s, t, i, val)
                    continue
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                stack.append((s, t, ~i))
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def query(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间和，与数组平方的区间和
        stack = [[s, t, i]]
        ans1 = ans2 = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans1 += self.cover[i]
                ans2 += self.cover_2[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return [ans1, ans2]


class SegmentTreePointChangeLongCon:
    def __init__(self, n) -> None:
        # 单点修改，查找最长的01交替字符子串连续区间
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.left_0 = [0] * (4 * self.n)
        self.left_1 = [0] * (4 * self.n)
        self.right_0 = [0] * (4 * self.n)
        self.right_1 = [0] * (4 * self.n)
        self.build()
        return

    @staticmethod
    def _max(a, b):
        return a if a > b else b

    @staticmethod
    def _min(a: int, b: int) -> int:
        return a if a < b else b

    def _push_up(self, i, s, m, t):
        # 合并区间的函数
        self.cover[i] = self._max(self.cover[2 * i], self.cover[2 * i + 1])
        self.cover[i] = self._max(self.cover[i], self.right_0[2 * i] + self.left_1[2 * i + 1])
        self.cover[i] = self._max(self.cover[i], self.right_1[2 * i] + self.left_0[2 * i + 1])

        self.left_0[i] = self.left_0[2 * i]
        if self.left_0[2 * i] == m - s + 1:
            self.left_0[i] += self.left_0[2 * i + 1] if (m - s + 1) % 2 == 0 else self.left_1[2 * i + 1]

        self.left_1[i] = self.left_1[2 * i]
        if self.left_1[2 * i] == m - s + 1:
            self.left_1[i] += self.left_1[2 * i + 1] if (m - s + 1) % 2 == 0 else self.left_0[2 * i + 1]

        self.right_0[i] = self.right_0[2 * i + 1]
        if self.right_0[2 * i + 1] == t - m:
            self.right_0[i] += self.right_0[2 * i] if (t - m) % 2 == 0 else self.right_1[2 * i]

        self.right_1[i] = self.right_1[2 * i + 1]
        if self.right_1[2 * i + 1] == t - m:
            self.right_1[i] += self.right_1[2 * i] if (t - m) % 2 == 0 else self.right_0[2 * i]
        return

    def build(self) -> None:
        # 数组初始化线段树
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = 1
                    self.left_0[i] = 1
                    self.right_0[i] = 1
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                stack.append((s, m, 2 * i))
                stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                m = s + (t - s) // 2
                self._push_up(i, s, m, t)
        return

    def update(self, left: int, right: int, s: int, t: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 增| val 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.left_0[i] = 1 - self.left_0[i]
                    self.right_0[i] = 1 - self.right_0[i]
                    self.left_1[i] = 1 - self.left_1[i]
                    self.right_1[i] = 1 - self.right_1[i]
                    self.cover[i] = 1
                    continue
                m = s + (t - s) // 2
                stack.append((s, t, ~i))
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                m = s + (t - s) // 2
                self._push_up(i, s, m, t)
        return

    def query(self):
        return self.cover[1]


class SegmentTreeRangeAndOrXOR:
    def __init__(self, n) -> None:
        # 区间修改成01或者翻转，区间查询最多有多少连续的1，以及总共有多少1
        self.n = n
        self.cover_1 = [0] * (4 * self.n)
        self.cover_0 = [0] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        self.left_1 = [0] * (4 * self.n)
        self.right_1 = [0] * (4 * self.n)
        self.left_0 = [0] * (4 * self.n)
        self.right_0 = [0] * (4 * self.n)
        self.lazy = [inf] * (4 * self.n)
        return

    @staticmethod
    def _max(a, b):
        return a if a > b else b

    @staticmethod
    def _min(a: int, b: int) -> int:
        return a if a < b else b

    def _push_up(self, i, s, m, t):
        # 合并区间的函数
        self.cover_1[i] = self._max(self.cover_1[2 * i], self.cover_1[2 * i + 1])
        self.cover_1[i] = self._max(self.cover_1[i], self.right_1[2 * i] + self.left_1[2 * i + 1])

        self.cover_0[i] = self._max(self.cover_0[2 * i], self.cover_0[2 * i + 1])
        self.cover_0[i] = self._max(self.cover_0[i], self.right_0[2 * i] + self.left_0[2 * i + 1])

        self.sum[i] = self.sum[2 * i] + self.sum[2 * i + 1]

        self.left_1[i] = self.left_1[2 * i]
        if self.left_1[i] == m - s + 1:
            self.left_1[i] += self.left_1[2 * i + 1]

        self.left_0[i] = self.left_0[2 * i]
        if self.left_0[i] == m - s + 1:
            self.left_0[i] += self.left_0[2 * i + 1]

        self.right_1[i] = self.right_1[2 * i + 1]
        if self.right_1[i] == t - m:
            self.right_1[i] += self.right_1[2 * i]

        self.right_0[i] = self.right_0[2 * i + 1]
        if self.right_0[i] == t - m:
            self.right_0[i] += self.right_0[2 * i]

        return

    def _make_tag(self, s, t, i, val):
        if val == 0:
            self.cover_1[i] = 0
            self.sum[i] = 0
            self.cover_0[i] = t - s + 1
            self.left_1[i] = self.right_1[i] = 0
            self.right_0[i] = self.left_0[i] = t - s + 1
        elif val == 1:
            self.cover_1[i] = t - s + 1
            self.cover_0[i] = 0
            self.sum[i] = t - s + 1
            self.left_1[i] = self.right_1[i] = t - s + 1
            self.right_0[i] = self.left_0[i] = 0
        else:
            self.cover_1[i], self.cover_0[i] = self.cover_0[i], self.cover_1[i]
            self.sum[i] = t - s + 1 - self.sum[i]
            self.left_0[i], self.left_1[i] = self.left_1[i], self.left_0[i]
            self.right_0[i], self.right_1[i] = self.right_1[i], self.right_0[i]
        self.lazy[i] = val
        return

    def _push_down(self, i, s, m, t):
        if self.lazy[i] != inf:
            self._make_tag(s, m, 2 * i, self.lazy[i])
            self._make_tag(m + 1, t, 2 * i + 1, self.lazy[i])
            self.lazy[i] = inf

    def build(self, nums) -> None:
        # 数组初始化线段树
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    if nums[s] == 0:
                        self.cover_1[i] = 0
                        self.sum[i] = 0
                        self.cover_0[i] = 1
                        self.left_1[i] = self.right_1[i] = 0
                        self.left_0[i] = self.right_0[i] = 1
                    else:
                        self.cover_1[i] = 1
                        self.sum[i] = 1
                        self.cover_0[i] = 0
                        self.left_1[i] = self.right_1[i] = 1
                        self.left_0[i] = self.right_0[i] = 0
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                stack.append((s, m, 2 * i))
                stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                m = s + (t - s) // 2
                self._push_up(i, s, m, t)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 增| val 而 i 从 1 开始，直接修改到底
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                m = s + (t - s) // 2
                if s < t:
                    self._push_down(i, s, m, t)
                if left <= s and t <= right:
                    self._make_tag(s, t, i, val)
                    continue

                stack.append((s, t, ~i))
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                m = s + (t - s) // 2
                self._push_up(i, s, m, t)
        return

    def query_sum(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间和，与数组平方的区间和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.sum[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans

    def query_max_length(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间的最大连续和，注意这里是递归
        if left <= s and t <= right:
            return self.cover_1[i], self.left_1[i], self.right_1[i]

        m = s + (t - s) // 2
        self._push_down(i, s, m, t)

        if right <= m:
            return self.query_max_length(left, right, s, m, 2 * i)
        if left > m:
            return self.query_max_length(left, right, m + 1, t, 2 * i + 1)

        res1 = self.query_max_length(left, right, s, m, 2 * i)
        res2 = self.query_max_length(left, right, m + 1, t, 2 * i + 1)

        # 参照区间递归方式
        res = [0] * 3
        res[0] = self._max(res1[0], res2[0])
        res[0] = self._max(res[0], res1[2] + res2[1])

        res[1] = res1[1]
        if res[1] == m - s + 1:
            res[1] += res2[1]

        res[2] = res2[2]
        if res[2] == t - m:
            res[2] += res1[2]
        return res


class SegmentTreeLongestSubSame:
    # 单点字母更新，最长具有相同字母的连续子数组查询
    def __init__(self, n, lst):
        self.n = n
        self.lst = lst
        self.pref = [0] * 4 * n
        self.suf = [0] * 4 * n
        self.ceil = [0] * 4 * n
        self.build()
        return

    def build(self):
        # 数组初始化线段树
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self._make_tag(ind)
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self._push_up(ind, s, t)
        return

    def _make_tag(self, i):
        # 只有此时 i 对应的区间 s==t 才打标记
        self.pref[i] = 1
        self.suf[i] = 1
        self.ceil[i] = 1
        return

    def _push_up(self, i, s, t):
        m = s + (t - s) // 2
        # 左右区间段分别为 [s, m] 与 [m+1, t] 保证 s < t
        self.pref[i] = self.pref[2 * i]
        if self.pref[2 * i] == m - s + 1 and self.lst[m] == self.lst[m + 1]:
            self.pref[i] += self.pref[2 * i + 1]

        self.suf[i] = self.suf[2 * i + 1]
        if self.suf[2 * i + 1] == t - m and self.lst[m] == self.lst[m + 1]:
            self.suf[i] += self.suf[2 * i]

        a = -inf
        for b in [self.pref[i], self.suf[i], self.ceil[2 * i], self.ceil[2 * i + 1]]:
            a = a if a > b else b
        if self.lst[m] == self.lst[m + 1]:
            b = self.suf[2 * i] + self.pref[2 * i + 1]
            a = a if a > b else b
        self.ceil[i] = a
        return

    def update_point(self, left, right, s, t, val, i):
        # 更新单点最小值
        self.lst[left] = val
        stack = []
        while True:
            stack.append([s, t, i])
            if left <= s <= t <= right:
                self._make_tag(i)
                break
            m = s + (t - s) // 2
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        stack.pop()
        while stack:
            s, t, i = stack.pop()
            self._push_up(i, s, t)
        assert i == 1
        # 获取当前最大的连续子串
        return self.ceil[1]


class SegmentTreeRangeXORQuery:
    def __init__(self, n) -> None:
        # 区间异或修改、单点异或查询
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        return

    def _push_up(self, i):
        # 合并区间的函数
        self.cover[i] = self.cover[2 * i] ^ self.cover[2 * i + 1]
        return

    def _make_tag(self, i, val):
        self.cover[i] ^= val
        self.lazy[i] ^= val
        return

    def _push_down(self, i):
        if self.lazy[i]:
            self._make_tag(2 * i, self.lazy[i])
            self._make_tag(2 * i + 1, self.lazy[i])
            self.lazy[i] = 0

    def update_range(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 异或 val 而 i 从 1 开始
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                m = s + (t - s) // 2
                if left <= s and t <= right:
                    self._make_tag(i, val)
                    continue
                self._push_down(i)
                stack.append((s, t, ~i))
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def update_point(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        # 修改点值 [left  right] 取值为 0 到 n-1 异或 val 而 i 从 1 开始
        assert left == right
        while True:
            if left <= s and t <= right:
                self._make_tag(i, val)
                break
            self._push_down(i)
            m = s + (t - s) // 2
            if left <= m:  # 注意左右子树的边界与范围
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        while i > 1:
            i //= 2
            self._push_up(i)
        return

    def query(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间和，与数组平方的区间和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans ^= self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans

    def query_point(self, left: int, right: int, s: int, t: int, i: int):
        # 查询区间和，与数组平方的区间和
        assert left == right
        ans = 0
        while True:
            if left <= s and t <= right:
                ans ^= self.cover[i]
                break
            m = s + (t - s) // 2
            self._push_down(i)
            if left <= m:
                s, t, i = s, m, 2 * i
            if right > m:
                s, t, i = m + 1, t, 2 * i + 1
        return ans


class SegmentTreeRangeSqrtSum:
    def __init__(self, n):
        # 区间值开方向下取整，区间和查询
        self.n = n
        self.cover = [0] * (4 * self.n)  # 区间和
        self.lazy = [inf] * (4 * self.n)  # 懒标记

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, ind = stack.pop()
            if ind >= 0:
                if s == t:
                    self.cover[ind] = nums[s]
                else:
                    stack.append([s, t, ~ind])
                    m = s + (t - s) // 2
                    stack.append([s, m, 2 * ind])
                    stack.append([m + 1, t, 2 * ind + 1])
            else:
                ind = ~ind
                self.cover[ind] = self.cover[2 * ind] + self.cover[2 * ind + 1]
        return

    def change(self, left, right, s, t, i):
        # 更新区间值
        stack = [[s, t, i]]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if self.cover[i] == t - s + 1:
                    continue
                if s == t:
                    self.cover[i] = int(self.cover[i] ** 0.5)
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                if left <= m:  # 注意左右子树的边界与范围
                    stack.append((s, m, 2 * i))
                if right > m:
                    stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query_sum(self, left, right, s, t, i):
        # 查询区间的和
        stack = [[s, t, i]]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, 2 * i))
            if right > m:
                stack.append((m + 1, t, 2 * i + 1))
        return ans
