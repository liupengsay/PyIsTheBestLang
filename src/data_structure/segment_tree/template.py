from collections import defaultdict
from typing import List

from src.utils.fast_io import inf


class RangeLongestRegularBrackets:
    def __init__(self, n) -> None:
        """query the longest regular brackets of static range"""
        self.n = n
        self.cover = [0] * (4 * n)
        self.left = [0] * (4 * n)
        self.right = [0] * (4 * n)
        return

    def build(self, brackets: str) -> None:
        assert self.n == len(brackets)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    if brackets[t] == "(":
                        self.left[i] = 1
                    else:
                        self.right[i] = 1
                    continue
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def _push_up(self, i):
        lst1 = (self.cover[i << 1], self.left[i << 1], self.right[i << 1])
        lst2 = (self.cover[(i << 1) | 1], self.left[(i << 1) | 1], self.right[(i << 1) | 1])
        self.cover[i], self.left[i], self.right[i] = self._merge_value(lst1, lst2)
        return

    @staticmethod
    def _merge_value(lst1, lst2):
        c1, left1, right1 = lst1[:]
        c2, left2, right2 = lst2[:]
        new = left1 if left1 < right2 else right2
        c = c1 + c2 + new * 2
        left = left1 + left2 - new
        right = right1 + right2 - new
        return c, left, right

    def range_longest_regular_brackets(self, left: int, right: int) -> int:
        stack = [(0, self.n - 1, 1)]
        ans = (0, 0, 0)
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                # merge params order attention
                ans = self._merge_value((self.cover[i], self.left[i], self.right[i]), ans)
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return list(ans)[0]


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
        self.cover[i] = self._max(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def _push_down(self, i):
        if self.lazy[i] != -inf:
            self.cover[i << 1] = self._max(self.cover[i << 1], self.lazy[i])
            self.cover[(i << 1) | 1] = self._max(self.cover[(i << 1) | 1], self.lazy[i])
            self.lazy[i << 1] = self._max(self.lazy[i << 1], self.lazy[i])
            self.lazy[(i << 1) | 1] = self._max(self.lazy[(i << 1) | 1], self.lazy[i])
            self.lazy[i] = -inf
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, nums[s])
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
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
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_ascend(self, left, right, val):
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
                    stack.append((a, m, i << 1))
                if right > m:
                    stack.append((m + 1, b, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_max(self, left, right):
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
                stack.append((a, m, i << 1))
            if right > m:
                stack.append((m + 1, b, (i << 1) | 1))
        return highest


class RangeAddRangeAvgDev:
    def __init__(self, n) -> None:
        """range_add|range_avg|range_dev"""
        self.n = n
        self.cover = [0] * (4 * self.n)  # x sum of range
        self.cover_square = [0] * (4 * self.n)  # x^2 sum of range
        self.lazy = [0] * (4 * self.n)
        return

    @staticmethod
    def _max(a, b):
        return a if a > b else b

    @staticmethod
    def _min(a: int, b: int) -> int:
        return a if a < b else b

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.cover_square[i] = self.cover_square[i << 1] + self.cover_square[(i << 1) | 1]
        return

    def _make_tag(self, i, s, t, val):
        self.cover_square[i] += self.cover[i] * 2 * val + (t - s + 1) * val * val
        self.cover[i] += val * (t - s + 1)
        self.lazy[i] += val
        return

    def _push_down(self, i, s, m, t):
        if self.lazy[i]:
            self._make_tag(i << 1, s, m, self.lazy[i])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy[i])
            self.lazy[i] = 0

    def build(self, nums) -> None:
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                    self.cover_square[i] = nums[s] * nums[s]
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
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
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_add(self, left: int, right: int, val: int) -> None:
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
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_avg_dev(self, left: int, right: int):
        stack = [(0, self.n - 1, 1)]
        ans1 = ans2 = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans1 += self.cover[i]
                ans2 += self.cover_square[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        avg = ans1 / (right - left + 1)
        dev = ans2 / (right - left + 1) - (ans1 / (right - left + 1)) ** 2
        return [avg, dev]


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
        self.cover[i] = self._max(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def _push_down(self, i):
        if self.lazy[i] != -inf:
            self.cover[i << 1] = self._max(self.cover[i << 1], self.lazy[i])
            self.cover[(i << 1) | 1] = self._max(self.cover[(i << 1) | 1], self.lazy[i])
            self.lazy[i << 1] = self._max(self.lazy[i << 1], self.lazy[i])
            self.lazy[(i << 1) | 1] = self._max(self.lazy[(i << 1) | 1], self.lazy[i])
            self.lazy[i] = -inf
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, nums[s])
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
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
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
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
                    stack.append((a, m, i << 1))
                if right > m:
                    stack.append((m + 1, b, (i << 1) | 1))
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
                stack.append((a, m, i << 1))
            if right > m:
                stack.append((m + 1, b, (i << 1) | 1))
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
            if right > m and self.cover[(i << 1) | 1] >= val:
                stack.append((m + 1, b, (i << 1) | 1))
            if left <= m and self.cover[i << 1] >= val:
                stack.append((a, m, i << 1))
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
        self.cover[i] = self._min(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def _push_down(self, i):
        if self.lazy[i] != inf:
            self.cover[i << 1] = self._min(self.cover[i << 1], self.lazy[i])
            self.cover[(i << 1) | 1] = self._min(self.cover[(i << 1) | 1], self.lazy[i])
            self.lazy[i << 1] = self._min(self.lazy[i << 1], self.lazy[i])
            self.lazy[(i << 1) | 1] = self._min(self.lazy[(i << 1) | 1], self.lazy[i])
            self.lazy[i] = inf
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, nums[s])
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
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
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
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
                    stack.append((a, m, i << 1))
                if right > m:
                    stack.append((m + 1, b, (i << 1) | 1))
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
                stack.append((a, m, i << 1))
            if right > m:
                stack.append((m + 1, b, (i << 1) | 1))
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
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, s, t, nums[s])
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def _push_down(self, i: int, s: int, m: int, t: int) -> None:
        if self.lazy[i]:
            self.cover[i << 1] += self.lazy[i] * (m - s + 1)
            self.cover[(i << 1) | 1] += self.lazy[i] * (t - m)

            self.floor[i << 1] += self.lazy[i]
            self.floor[(i << 1) | 1] += self.lazy[i]

            self.ceil[i << 1] += self.lazy[i]
            self.ceil[(i << 1) | 1] += self.lazy[i]

            self.lazy[i << 1] += self.lazy[i]
            self.lazy[(i << 1) | 1] += self.lazy[i]

            self.lazy[i] = 0

    def _push_up(self, i) -> None:
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.ceil[i] = self._max(self.ceil[i << 1], self.ceil[(i << 1) | 1])
        self.floor[i] = self._min(self.floor[i << 1], self.floor[(i << 1) | 1])
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
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
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
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums


class RangeAddMulRangeSum:

    def __init__(self, n, mod):
        self.n = n
        self.mod = mod
        self.cover = [0] * (4 * n)
        self.add = [0] * (4 * n)  # lazy_tag for mul
        self.mul = [1] * (4 * n)  # lazy_tag for add
        return

    def _make_tag(self, i, s, t, val, op="add") -> None:
        if op == "add":
            self.cover[i] = (self.cover[i] + (t - s + 1) * val) % self.mod
            self.add[i] = (self.add[i] + val) % self.mod
        else:
            self.cover[i] = (self.cover[i] * val) % self.mod
            self.add[i] = (self.add[i] * val) % self.mod
            self.mul[i] = (self.mul[i] * val) % self.mod
        return

    def _push_up(self, i):
        self.cover[i] = (self.cover[i << 1] + self.cover[(i << 1) | 1]) % self.mod
        return

    def _push_down(self, i, s, m, t):
        self.cover[i << 1] = (self.cover[i << 1] * self.mul[i] + self.add[i] * (m - s + 1)) % self.mod
        self.cover[(i << 1) | 1] = (self.cover[(i << 1) | 1] * self.mul[i] + self.add[i] * (t - m)) % self.mod

        self.mul[i << 1] = (self.mul[i << 1] * self.mul[i]) % self.mod
        self.mul[(i << 1) | 1] = (self.mul[(i << 1) | 1] * self.mul[i]) % self.mod

        self.add[i << 1] = (self.add[i << 1] * self.mul[i] + self.add[i]) % self.mod
        self.add[(i << 1) | 1] = (self.add[(i << 1) | 1] * self.mul[i] + self.add[i]) % self.mod

        self.mul[i] = 1
        self.add[i] = 0
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, s, t, nums[s], "add")
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i] % self.mod
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_add_mul(self, left, right, val, op="add"):
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, s, t, val, op)
                    continue
                stack.append([s, t, ~i])
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_sum(self, left: int, right: int) -> int:
        # query the range sum
        if left == right:
            s, t, i = 0, self.n - 1, 1
            ans = 0
            while True:
                if left <= s <= t <= right:
                    ans += self.cover[i] % self.mod
                    break
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                if left <= m:
                    s, t, i = s, m, i << 1
                if right > m:
                    s, t, i = m + 1, t, (i << 1) | 1
            return ans

        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                ans %= self.mod
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class RangeAffineRangeSum:

    def __init__(self, n, mod, m=32):
        self.n = n
        self.mod = mod
        self.m = m
        self.mul = 1 << self.m
        self.mask = (1 << self.m) - 1
        self.cover = [0] * (4 * n)
        self.tag = [self.mul] * (4 * n)
        return

    def _make_tag(self, i, s, t, val) -> None:
        mul, add = val >> self.m, val & self.mask
        self.cover[i] = (self.cover[i] * mul + (t - s + 1) * add) % self.mod
        self.tag[i] = self._combine_tag(self.tag[i], val)
        return

    def _combine_tag(self, x1, x2):
        mul1, add1 = x1 >> self.m, x1 & self.mask
        mul2, add2 = x2 >> self.m, x2 & self.mask
        mul = (mul2 * mul1) % self.mod
        add = (mul2 * add1 + add2) % self.mod
        return (mul << self.m) | add

    def _push_up(self, i):
        self.cover[i] = (self.cover[i << 1] + self.cover[(i << 1) | 1]) % self.mod
        return

    def _push_down(self, i, s, m, t):
        val = self.tag[i]
        if val != self.mul:
            self._make_tag(i << 1, s, m, val)
            self._make_tag((i << 1) | 1, m + 1, t, val)
            self.tag[i] = self.mul
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, s, t, nums[s])
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i] % self.mod
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_affine(self, left, right, val):
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, s, t, val)
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_sum(self, left: int, right: int) -> int:
        if left == right:
            s, t, i = 0, self.n - 1, 1
            ans = 0
            while True:
                if left <= s <= t <= right:
                    ans += self.cover[i] % self.mod
                    break
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                if left <= m:
                    s, t, i = s, m, i << 1
                if right > m:
                    s, t, i = m + 1, t, (i << 1) | 1
            return ans

        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                ans %= self.mod
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


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
            self.cover[i << 1] = self.lazy[i] * (m - s + 1)
            self.cover[(i << 1) | 1] = self.lazy[i] * (t - m)

            self.floor[i << 1] = self.lazy[i]
            self.floor[(i << 1) | 1] = self.lazy[i]

            self.ceil[i << 1] = self.lazy[i]
            self.ceil[(i << 1) | 1] = self.lazy[i]

            self.lazy[i << 1] = self.lazy[i]
            self.lazy[(i << 1) | 1] = self.lazy[i]

            self.lazy[i] = inf

    def _push_up(self, i) -> None:
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.ceil[i] = self._max(self.ceil[i << 1], self.ceil[(i << 1) | 1])
        self.floor[i] = self._min(self.floor[i << 1], self.floor[(i << 1) | 1])
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
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, s, t, nums[s])
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
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
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
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
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
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
            self.cover[i << 1] = self.lazy[i] * (m - s + 1)
            self.cover[(i << 1) | 1] = self.lazy[i] * (t - m)

            self.floor[i << 1] = self.lazy[i]
            self.floor[(i << 1) | 1] = self.lazy[i]

            self.ceil[i << 1] = self.lazy[i]
            self.ceil[(i << 1) | 1] = self.lazy[i]

            self.lazy[i << 1] = self.lazy[i]
            self.lazy[(i << 1) | 1] = self.lazy[i]

            self.lazy[i] = inf

    def _push_up(self, i) -> None:
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.ceil[i] = self._max(self.ceil[i << 1], self.ceil[(i << 1) | 1])
        self.floor[i] = self._min(self.floor[i << 1], self.floor[(i << 1) | 1])
        return

    def _make_tag(self, i, s, t, val) -> None:
        self.cover[i] = val * (t - s + 1)
        self.floor[i] = val
        self.ceil[i] = val
        self.lazy[i] = val
        return

    def range_change(self, left, right, val):
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
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return highest


class RangeKthSmallest:
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
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i].append(nums[s])
                    continue
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def _range_merge_to_disjoint(self, lst1, lst2):
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
        self.cover[i] = self._range_merge_to_disjoint(self.cover[i << 1][:], self.cover[(i << 1) | 1][:])
        return

    def range_kth_smallest(self, left: int, right: int) -> int:
        stack = [(0, self.n - 1, 1)]
        ans = []
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = self._range_merge_to_disjoint(ans, self.cover[i][:])
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
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
            self.cover[i << 1] |= self.lazy[i]
            self.cover[(i << 1) | 1] |= self.lazy[i]

            self.lazy[i << 1] |= self.lazy[i]
            self.lazy[(i << 1) | 1] |= self.lazy[i]

            self.lazy[i] = 0

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] & self.cover[(i << 1) | 1]
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, nums[s])
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
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
                    stack.append((s, m, i << 1))
                if r > m:
                    stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, m, i << 1))
            if r > m:
                stack.append((m + 1, t, (i << 1) | 1))
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
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums


class RangeRevereRangeBitCount:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        return

    def build(self, nums) -> None:
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        return

    def _push_down(self, i: int, s: int, m: int, t: int) -> None:
        if self.lazy[i]:
            self.cover[i << 1] = m - s + 1 - self.cover[i << 1]
            self.cover[(i << 1) | 1] = t - m - self.cover[(i << 1) | 1]

            self.lazy[i << 1] ^= self.lazy[i]
            self.lazy[(i << 1) | 1] ^= self.lazy[i]

            self.lazy[i] = 0
        return

    def range_reverse(self, left: int, right: int) -> None:
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = t - s + 1 - self.cover[i]
                    self.lazy[i] ^= 1
                    continue

                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                stack.append((s, t, ~i))

                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        return

    def range_bit_count(self, left: int, right: int) -> int:
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class RangeChangeRangeOr:
    def __init__(self, n) -> None:
        self.n = n
        self.lazy = [inf] * (4 * self.n)
        self.cover = [0] * (4 * self.n)
        return

    def _make_tag(self, val: int, i: int) -> None:
        self.cover[i] = val
        self.lazy[i] = val
        return

    def _push_down(self, i: int) -> None:
        if self.lazy[i] != inf:
            self._make_tag(self.lazy[i], i << 1)
            self._make_tag(self.lazy[i], (i << 1) | 1)
            self.lazy[i] = inf

    def _push_up(self, i: int) -> None:
        self.cover[i] = self.cover[i << 1] | self.cover[(i << 1) | 1]

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(nums[s], i)
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
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
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_change(self, left: int, right: int, val: int) -> None:
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
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_or(self, left: int, right: int) -> int:
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
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class SegmentTreePointUpdateRangeMulQuery:
    def __init__(self, n, mod) -> None:
        """range_change|range_mul"""
        self.n = n
        self.mod = mod
        self.cover = [1] * (4 * self.n)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        stack = [(s, t, i)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = val
                    continue
                m = s + (t - s) // 2
                stack.append((s, t, ~i))
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self.cover[i] = self.cover[i << 1] * self.cover[(i << 1) | 1]
                self.cover[i] %= self.mod
        return

    def query_mul(self, left: int, right: int, s: int, t: int, i: int) -> int:
        stack = [(s, t, i)]
        ans = 1
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans *= self.cover[i]
                ans %= self.mod
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class RangeChangeAddRangeMax:

    def __init__(self, n):
        self.n = n
        self.inf = 1 << 68
        self.cover = [0] * (4 * n)
        self.tag = [1] * (4 * n)
        return

    def _make_tag(self, i, val) -> None:
        res = self.mask_to_value(val)
        if res[0] > -self.inf:
            self.cover[i] = res[0]
        else:
            self.cover[i] += res[1]
        self.tag[i] = self._combine_tag(val, self.tag[i])
        return

    @staticmethod
    def add_to_mask(val):
        if val >= 0:
            return (val << 2) | 2 | 1
        return ((-val) << 2) | 1

    @staticmethod
    def change_to_mask(val):
        if val >= 0:
            return (val << 2) | 2
        return (-val) << 2

    def mask_to_value(self, val):
        res = [-self.inf, -self.inf]
        if not val & 1:
            res[0] = (val >> 2) * (2 * ((val & 2) >> 1) - 1)
        else:
            res[1] = (val >> 2) * (2 * ((val & 2) >> 1) - 1)
        return res

    def _combine_tag(self, val1, val2):
        res1 = self.mask_to_value(val1)
        res2 = self.mask_to_value(val2)
        if res1[0] > -self.inf:
            return self.change_to_mask(res1[0])
        if res2[0] > -self.inf:
            res2[0] += res1[1]
            return self.change_to_mask(res2[0])
        res2[1] += res1[1]
        return self.add_to_mask(res2[1])

    def _push_up(self, i):
        left, right = self.cover[i << 1], self.cover[(i << 1) | 1]
        self.cover[i] = left if left > right else right
        return

    def _push_down(self, i):
        val = self.tag[i]
        if val > 1:
            self._make_tag(i << 1, val)
            self._make_tag((i << 1) | 1, val)
            self.tag[i] = 1
        return

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
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
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_change_add(self, left, right, val):
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, val)
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                self._push_down(i)
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_max(self, left: int, right: int) -> int:
        ans = -self.inf
        if left == right:
            s, t, i = 0, self.n - 1, 1
            while True:
                if left <= s <= t <= right:
                    ans = ans if ans > self.cover[i] else self.cover[i]
                    break
                m = s + (t - s) // 2
                self._push_down(i)
                if left <= m:
                    s, t, i = s, m, i << 1
                if right > m:
                    s, t, i = m + 1, t, (i << 1) | 1
            return ans

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = ans if ans > self.cover[i] else self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class PointSetRangeMaxNonEmpConSubSum:
    def __init__(self, n) -> None:
        self.n = n
        self.cover = [-inf] * (4 * self.n)
        self.left = [-inf] * (4 * self.n)
        self.right = [-inf] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        return

    @staticmethod
    def max(a, b):
        return a if a > b else b

    def _merge_value(self, res1, res2):
        res = [0] * 4
        res[0] = self.max(res1[0], res2[0])
        res[0] = self.max(res[0], res1[2] + res2[1])
        res[1] = self.max(res1[1], res1[3] + res2[1])
        res[2] = self.max(res2[2], res2[3] + res1[2])
        res[3] = res1[3] + res2[3]
        return res

    def _push_up(self, i):
        res1 = [self.cover[i << 1], self.left[i << 1], self.right[i << 1], self.sum[i << 1]]
        res2 = [self.cover[(i << 1) | 1], self.left[(i << 1) | 1], self.right[(i << 1) | 1], self.sum[(i << 1) | 1]]
        self.cover[i], self.left[i], self.right[i], self.sum[i] = self._merge_value(res1, res2)
        return

    def build(self, nums) -> None:
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                    self.left[i] = nums[s]
                    self.right[i] = nums[s]
                    self.sum[i] = nums[s]
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def point_set(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = val
                    self.left[i] = val
                    self.right[i] = val
                    self.sum[i] = val
                    continue
                m = s + (t - s) // 2
                stack.append((s, t, ~i))
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_max_non_emp_con_sub_sum(self, left: int, right: int):
        stack = [(0, self.n - 1, 1)]
        dct = defaultdict(lambda: [-inf] * 4)
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    dct[i] = [self.cover[i], self.left[i], self.right[i], self.sum[i]]
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2

                if right <= m:
                    stack.append((s, m, 2 * i))
                    continue
                if left > m:
                    stack.append((m + 1, t, 2 * i + 1))
                    continue
                stack.append((s, m, 2 * i))
                stack.append((m + 1, t, 2 * i + 1))
            else:
                i = ~i
                dct[i] = self._merge_value(dct[i << 1], dct[(i << 1) | 1])
        return dct[1][0]


class PointSetRangeComposite:

    def __init__(self, n, mod, m=32):
        self.n = n
        self.mod = mod
        self.m = m
        self.mask = (1 << m) - 1
        self.cover = [0] * (4 * n)
        return

    def _make_tag(self, i, val) -> None:
        self.cover[i] = val
        return

    def _push_up(self, i):
        self.cover[i] = self._merge_cover(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def _merge_cover(self, val1, val2):
        mul1, add1 = val1 >> self.m, val1 & self.mask
        mul2, add2 = val2 >> self.m, val2 & self.mask
        return ((mul2 * mul1 % self.mod) << self.m) | ((mul2 * add1 + add2) % self.mod)

    def build(self, nums: List[int]) -> None:
        assert self.n == len(nums)
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, nums[s])
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                val = self.cover[i]
                nums[s] = (val >> self.m) + (val & self.mask)
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, left, right, val):
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, val)
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_composite(self, left: int, right: int) -> int:
        stack = [(0, self.n - 1, 1)]
        ans = 1 << self.m
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = self._merge_cover(self.cover[i], ans)
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class RangeChangeRangeMaxNonEmpConSubSum:
    def __init__(self, n, initial=inf) -> None:
        """range_change|range_max_con_sub_sum"""
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
            self._make_tag(i << 1, self.lazy[i])
            self._make_tag((i << 1) | 1, self.lazy[i])
            self.lazy[i] = self.initial
        return

    def _range_merge_to_disjoint(self, res1, res2):
        res = [0] * 4
        res[0] = self._max(res1[0], res2[0])
        res[0] = self._max(res[0], res1[2] + res2[1])
        res[1] = self._max(res1[1], res1[3] + res2[1])
        res[2] = self._max(res2[2], res2[3] + res1[2])
        res[3] = res1[3] + res2[3]
        return res

    def _push_up(self, i):
        res1 = self.cover[i << 1], self.left[i << 1], self.right[i << 1], self.sum[i << 1]
        res2 = self.cover[(i << 1) | 1], self.left[(i << 1) | 1], self.right[(i << 1) | 1], self.sum[(i << 1) | 1]
        self.cover[i], self.left[i], self.right[i], self.sum[i] = self._range_merge_to_disjoint(res1, res2)
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
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
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
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                self._push_down(i)
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
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
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
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
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                dct[i] = self._range_merge_to_disjoint(dct[i << 1], dct[(i << 1) | 1])
        return dct[1]


class SegmentTreePointChangeLongCon:
    def __init__(self, n) -> None:
        """point_change and 01_con_sub"""
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
        self.cover[i] = self._max(self.cover[i << 1], self.cover[(i << 1) | 1])
        self.cover[i] = self._max(self.cover[i], self.right_0[i << 1] + self.left_1[(i << 1) | 1])
        self.cover[i] = self._max(self.cover[i], self.right_1[i << 1] + self.left_0[(i << 1) | 1])

        self.left_0[i] = self.left_0[i << 1]
        if self.left_0[i << 1] == m - s + 1:
            self.left_0[i] += self.left_0[(i << 1) | 1] if (m - s + 1) % 2 == 0 else self.left_1[(i << 1) | 1]

        self.left_1[i] = self.left_1[i << 1]
        if self.left_1[i << 1] == m - s + 1:
            self.left_1[i] += self.left_1[(i << 1) | 1] if (m - s + 1) % 2 == 0 else self.left_0[(i << 1) | 1]

        self.right_0[i] = self.right_0[(i << 1) | 1]
        if self.right_0[(i << 1) | 1] == t - m:
            self.right_0[i] += self.right_0[i << 1] if (t - m) % 2 == 0 else self.right_1[i << 1]

        self.right_1[i] = self.right_1[(i << 1) | 1]
        if self.right_1[(i << 1) | 1] == t - m:
            self.right_1[i] += self.right_1[i << 1] if (t - m) % 2 == 0 else self.right_0[i << 1]
        return

    def build(self) -> None:
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
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                m = s + (t - s) // 2
                self._push_up(i, s, m, t)
        return

    def update(self, left: int, right: int, s: int, t: int, i: int) -> None:
        stack = [(s, t, i)]
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
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                m = s + (t - s) // 2
                self._push_up(i, s, m, t)
        return

    def query(self):
        return self.cover[1]


class RangeXorUpdateRangeXorQuery:
    def __init__(self, n) -> None:
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        return

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] ^ self.cover[(i << 1) | 1]
        return

    def _make_tag(self, i, s, t, val):
        if (t - s + 1) % 2:
            self.cover[i] ^= val
        self.lazy[i] ^= val
        return

    def _push_down(self, i, s, m, t):
        if self.lazy[i]:
            self._make_tag(i << 1, s, m, self.lazy[i])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy[i])
            self.lazy[i] = 0

    def build(self, nums) -> None:
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
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
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_xor_update(self, left: int, right: int, val: int) -> None:
        if left == right:
            assert 0 <= left == right <= self.n - 1
            s, t, i = 0, self.n - 1, 1
            while True:
                if left <= s and t <= right:
                    self._make_tag(i, s, t, val)
                    break
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                if left <= m:
                    s, t, i = s, m, i << 1
                if right > m:
                    s, t, i = m + 1, t, (i << 1) | 1
            while i > 1:
                i //= 2
                self._push_up(i)
            return

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                m = s + (t - s) // 2
                if left <= s and t <= right:
                    self._make_tag(i, s, t, val)
                    continue
                self._push_down(i, s, m, t)
                stack.append((s, t, ~i))
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_xor_query(self, left: int, right: int):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans ^= self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class SegmentTreeRangeAndOrXOR:
    def __init__(self, n) -> None:
        """range_change|range_reverse|range_con_sub|range_sum"""
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
        self.cover_1[i] = self._max(self.cover_1[i << 1], self.cover_1[(i << 1) | 1])
        self.cover_1[i] = self._max(self.cover_1[i], self.right_1[i << 1] + self.left_1[(i << 1) | 1])

        self.cover_0[i] = self._max(self.cover_0[i << 1], self.cover_0[(i << 1) | 1])
        self.cover_0[i] = self._max(self.cover_0[i], self.right_0[i << 1] + self.left_0[(i << 1) | 1])

        self.sum[i] = self.sum[i << 1] + self.sum[(i << 1) | 1]

        self.left_1[i] = self.left_1[i << 1]
        if self.left_1[i] == m - s + 1:
            self.left_1[i] += self.left_1[(i << 1) | 1]

        self.left_0[i] = self.left_0[i << 1]
        if self.left_0[i] == m - s + 1:
            self.left_0[i] += self.left_0[(i << 1) | 1]

        self.right_1[i] = self.right_1[(i << 1) | 1]
        if self.right_1[i] == t - m:
            self.right_1[i] += self.right_1[i << 1]

        self.right_0[i] = self.right_0[(i << 1) | 1]
        if self.right_0[i] == t - m:
            self.right_0[i] += self.right_0[i << 1]

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
            self._make_tag(s, m, i << 1, self.lazy[i])
            self._make_tag(m + 1, t, (i << 1) | 1, self.lazy[i])
            self.lazy[i] = inf

    def build(self, nums) -> None:
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
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                m = s + (t - s) // 2
                self._push_up(i, s, m, t)
        return

    def update(self, left: int, right: int, s: int, t: int, val: int, i: int) -> None:
        stack = [(s, t, i)]
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
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                m = s + (t - s) // 2
                self._push_up(i, s, m, t)
        return

    def query_sum(self, left: int, right: int, s: int, t: int, i: int):
        stack = [(s, t, i)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.sum[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def query_max_length(self, left: int, right: int, s: int, t: int, i: int):
        if left <= s and t <= right:
            return self.cover_1[i], self.left_1[i], self.right_1[i]

        m = s + (t - s) // 2
        self._push_down(i, s, m, t)

        if right <= m:
            return self.query_max_length(left, right, s, m, i << 1)
        if left > m:
            return self.query_max_length(left, right, m + 1, t, (i << 1) | 1)

        res1 = self.query_max_length(left, right, s, m, i << 1)
        res2 = self.query_max_length(left, right, m + 1, t, (i << 1) | 1)

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


class PointSetRangeLongestSubSame:
    def __init__(self, n, lst):
        self.n = n
        self.lst = lst
        self.pref = [0] * 4 * n
        self.suf = [0] * 4 * n
        self.cover = [0] * 4 * n
        self._build()
        return

    def _build(self):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i)
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i, s, t)
        return

    def _make_tag(self, i):
        self.pref[i] = 1
        self.suf[i] = 1
        self.cover[i] = 1
        return

    def _push_up(self, i, s, t):
        m = s + (t - s) // 2
        self.pref[i] = self.pref[i << 1]
        if self.pref[i << 1] == m - s + 1 and self.lst[m] == self.lst[m + 1]:
            self.pref[i] += self.pref[(i << 1) | 1]

        self.suf[i] = self.suf[(i << 1) | 1]
        if self.suf[(i << 1) | 1] == t - m and self.lst[m] == self.lst[m + 1]:
            self.suf[i] += self.suf[i << 1]

        a = -inf
        for b in [self.pref[i], self.suf[i], self.cover[i << 1], self.cover[(i << 1) | 1]]:
            a = a if a > b else b
        if self.lst[m] == self.lst[m + 1]:
            b = self.suf[i << 1] + self.pref[(i << 1) | 1]
            a = a if a > b else b
        self.cover[i] = a
        return

    def point_set_rang_longest_sub_same(self, x, val):
        self.lst[x] = val
        stack = []
        s, t, i = 0, self.n - 1, 1
        while True:
            stack.append((s, t, i))
            if s == t == x:
                self._make_tag(i)
                break
            m = s + (t - s) // 2
            if x <= m:
                s, t, i = s, m, i << 1
            if x > m:
                s, t, i = m + 1, t, (i << 1) | 1
        stack.pop()
        while stack:
            s, t, i = stack.pop()
            self._push_up(i, s, t)
        assert i == 1
        return self.cover[1]





class SegmentTreeRangeSqrtSum:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.lazy = [inf] * (4 * self.n)

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        return

    def change(self, left, right, s, t, i):
        stack = [(s, t, i)]
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
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        return

    def query_sum(self, left, right, s, t, i):
        stack = [(s, t, i)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i]
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans
