from collections import defaultdict

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

    def build(self, nums) -> None:
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

    def build(self, nums) -> None:
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

    def build(self, nums) -> None:
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

    def build(self, nums) -> None:
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

    def range_min_bisect_left(self, val) -> int:
        # query the range min with bisect_left
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if self.floor[i << 1] <= val:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return t

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

    def build(self, nums) -> None:
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

    def build(self, nums) -> None:
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


class RangeChangeReverseRangeSumLongestConSub:
    def __init__(self, n):
        """range_change|range_reverse|range_con_sub|range_sum"""
        self.n = n
        # cover_0, cover_1, sum_1, start_0, start_1, end_0, end_1, s1, t2
        self.cover = [(0,) * 9] * (4 * self.n)  # cover with 0
        self.lazy = [3] * (4 * self.n)  # lazy tag 0-change 1-change 2-reverse
        return

    @staticmethod
    def _merge_value(res1, res2):
        cover_01, cover_11, sum_11, start_01, start_11, end_01, end_11, s1, t1 = res1[:]
        cover_02, cover_12, sum_12, start_02, start_12, end_02, end_12, s2, t2 = res2[:]
        cover_0 = cover_01 if cover_01 > cover_02 else cover_02
        cover_0 = cover_0 if cover_0 > end_01 + start_02 else end_01 + start_02
        cover_1 = cover_11 if cover_11 > cover_12 else cover_12
        cover_1 = cover_1 if cover_1 > end_11 + start_12 else end_11 + start_12

        sum_1 = sum_11 + sum_12

        if start_01 == t1 - s1 + 1:
            start_0 = start_01 + start_02
        else:
            start_0 = start_01

        if start_11 == t1 - s1 + 1:
            start_1 = start_11 + start_12
        else:
            start_1 = start_11

        if end_02 == t2 - s2 + 1:
            end_0 = end_02 + end_01
        else:
            end_0 = end_02

        if end_12 == t2 - s2 + 1:
            end_1 = end_12 + end_11
        else:
            end_1 = end_12
        return cover_0, cover_1, sum_1, start_0, start_1, end_0, end_1, s1, t2

    def _push_up(self, i):
        self.cover[i] = self._merge_value(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def _make_tag(self, s, t, i, val):
        if val == 0:
            self.cover[i] = (t - s + 1, 0, 0, t - s + 1, 0, t - s + 1, 0, s, t)
        elif val == 1:
            self.cover[i] = (0, t - s + 1, t - s + 1, 0, t - s + 1, 0, t - s + 1, s, t)
        elif val == 2:  # 2
            cover_0, cover_1, sum_1, start_0, start_1, end_0, end_1, s, t = self.cover[i]
            self.cover[i] = (cover_1, cover_0, t - s + 1 - sum_1, start_1, start_0, end_1, end_0, s, t)
        tag = self.lazy[i]
        if val <= 1:
            self.lazy[i] = val
        else:
            if tag <= 1:
                tag = 1 - tag
            elif tag == 2:
                tag = 3
            else:
                tag = 2
            self.lazy[i] = tag
        return

    def _push_down(self, i, s, m, t):
        if self.lazy[i] != 3:
            self._make_tag(s, m, i << 1, self.lazy[i])
            self._make_tag(m + 1, t, (i << 1) | 1, self.lazy[i])
            self.lazy[i] = 3

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(s, t, i, nums[s])
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
                nums[s] = self.cover[i][2]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_change_reverse(self, left, right, val):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(s, t, i, val)
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
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover[i][2]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def range_longest_con_sub(self, left, right):
        ans = tuple()
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                cur = self.cover[i]
                if not ans:
                    ans = cur
                else:
                    ans = self._merge_value(cur, ans)
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans[1]


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

    def build(self, nums) -> None:
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

    def build(self, nums) -> None:
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

    def build(self, nums) -> None:
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


class PointSetRangeOr:

    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * n)
        return

    def _make_tag(self, i, val) -> None:
        self.cover[i] = val
        return

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] | self.cover[(i << 1) | 1]
        return

    def build(self, nums) -> None:
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
                nums[s] = val
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, left, right, val):
        assert 0 <= left == right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t == left:
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

    def range_or(self, left: int, right: int) -> int:
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans |= self.cover[i]
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def range_or_binary_search_right(self, left: int, k) -> int:
        stack = [(0, self.n - 1, 1)]
        ans = val = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and (self.cover[i] | val).bit_count() <= k:
                val |= self.cover[i]
                ans = t
                continue
            elif s == t:
                break
            else:
                m = s + (t - s) // 2
                stack.append((m + 1, t, (i << 1) | 1))
                if left <= m:
                    stack.append((s, m, i << 1))
        return ans


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

    def build(self, nums) -> None:
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
        assert 0 <= left == right <= self.n - 1
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

    def build(self, nums) -> None:
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

    def get(self):
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

    def range_max_non_emp_con_sub_sum(self, left: int, right: int):
        assert 0 <= left <= right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        ans = [-self.initial] * 4
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                cur = [self.cover[i], self.left[i], self.right[i], self.sum[i]]
                ans = self._range_merge_to_disjoint(cur, ans)
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans[0]


class PointSetRangeLongestAlter:
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

    def point_set_range_longest_alter(self, left: int, right: int) -> None:
        stack = [(0, self.n - 1, 1)]
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
        return self.cover[1]


class PointSetRangeMax:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover = [initial] * (4 * n)
        return

    @classmethod
    def merge(cls, x, y):
        return x if x > y else y

    def _push_up(self, i):
        self.cover[i] = self.merge(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]  # assert self.n == len(nums)
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
                val = self.cover[i]
                nums[s] = val
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1  # assert 0 <= left == right <= self.n - 1
        while True:
            if s == t == ind:
                self.cover[i] = val
                break
            m = s + (t - s) // 2
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        while i > 1:
            i //= 2
            self._push_up(i)
        return

    def range_max(self, left: int, right: int) -> int:
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = self.merge(ans, self.cover[i])
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def range_max_bisect_left(self, val) -> int:
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            if self.cover[i << 1] >= val:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return t


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


class PointSetRangeSum:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover = [initial] * (4 * n)
        return

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        return

    def build(self, nums) -> None:
        stack = [(0, self.n - 1, 1)]  # assert self.n == len(nums)
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
                val = self.cover[i]
                nums[s] = val
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1  # assert 0 <= left == right <= self.n - 1
        while True:
            if s == t == ind:
                self.cover[i] = val
                break
            m = s + (t - s) // 2
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        while i > 1:
            i //= 2
            self._push_up(i)
        return

    def range_sum(self, left: int, right: int) -> int:
        stack = [(0, self.n - 1, 1)]
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

    def range_sum_bisect_left(self, val) -> int:
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            if self.cover[i << 1] >= val:
                s, t, i = s, m, i << 1
            else:
                val -= self.cover[i << 1]
                s, t, i = m + 1, t, (i << 1) | 1
        return t


class PointSetRangeMin:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover = [initial] * (4 * n)
        return

    def _make_tag(self, i, val) -> None:
        self.cover[i] = val
        return

    @classmethod
    def _min(cls, x, y):
        return x if x < y else y

    def _push_up(self, i):
        self.cover[i] = self._min(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def build(self, nums) -> None:
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
                nums[s] = val
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, left, right, val):
        assert 0 <= left == right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t == left:
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

    def range_min(self, left: int, right: int) -> int:
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = self._min(ans, self.cover[i])
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class PointSetRangeMinCount:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.m = n.bit_length()
        self.mask = (1 << self.m) - 1
        self.cover = [initial] * (4 * n)
        return

    def _make_tag(self, i, val) -> None:
        self.cover[i] = (val << self.m) | 1
        return

    def _merge(self, a1, a2):
        x1, c1 = a1 >> self.m, a1 & self.mask
        x2, c2 = a2 >> self.m, a2 & self.mask
        if x1 < x2:
            return (x1 << self.m) | c1
        if x1 == x2:
            return (x1 << self.m) | (c1 + c2)
        return (x2 << self.m) | c2

    def _push_up(self, i):
        self.cover[i] = self._merge(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def build(self, nums) -> None:
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
                val = self.cover[i] >> self.m
                nums[s] = val
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, left, right, val):
        assert 0 <= left == right <= self.n - 1
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t == left:
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

    def range_min_count(self, left: int, right: int):
        stack = [(0, self.n - 1, 1)]
        ans = (inf << self.m) | 1
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = self._merge(ans, self.cover[i])
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans >> self.m, ans & self.mask


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


class RangeSqrtRangeSum:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * self.n)

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

    def range_sqrt(self, left, right):
        stack = [(0, self.n - 1, 1)]
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

    def range_sum(self, left, right):
        stack = [(0, self.n - 1, 1)]
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


class PointSetRangeMaxSubSum:
    def __init__(self, n, initial=inf):
        self.n = n
        self.initial = initial
        self.cover = [-initial] * (4 * self.n)
        self.left = [-initial] * (4 * self.n)
        self.right = [-initial] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        return

    def build(self, nums):
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

    @staticmethod
    def _max(a, b):
        return a if a > b else b

    def _make_tag(self, i, val):
        self.cover[i] = val
        self.left[i] = val
        self.right[i] = val
        self.sum[i] = val
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

    def point_set_range_max_sub_sum(self, ind, val):
        s, t, i = 0, self.n - 1, 1  # assert 0 <= ind <= self.n - 1
        while True:
            if s == t == ind:
                self._make_tag(i, val)
                break
            m = s + (t - s) // 2
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        while i > 1:
            i //= 2
            self._push_up(i)
        return self.cover[1]
