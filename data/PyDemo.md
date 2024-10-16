

class BinarySearch:
    def __init__(self):
        return

    @staticmethod
    def find_int_left(low: int, high: int, check) -> int:
        """find the minimum int x which make check true"""
        while low < high:
            mid = low + (high - low) // 2
            if check(mid):
                high = mid
            else:
                low = mid + 1
        return low

    @staticmethod
    def find_int_right(low: int, high: int, check) -> int:
        """find the maximum int x which make check true"""
        while low < high:
            mid = low + (high - low + 1) // 2
            if check(mid):
                low = mid
            else:
                high = mid - 1
        return high

    @staticmethod
    def find_float_left(low: float, high: float, check, error=1e-6) -> float:
        """find the minimum float x which make check true"""
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                high = mid
            else:
                low = mid
        return low if check(low) else high

    @staticmethod
    def find_float_right(low: float, high: float, check, error=1e-6) -> float:
        """find the maximum float x which make check true"""
        while low < high - error:
            mid = low + (high - low) / 2
            if check(mid):
                low = mid
            else:
                high = mid
        return high if check(high) else low
class CircleSection:
    def __init__(self):
        return

    @staticmethod
    def compute_circle_result(n: int, m: int, x: int, tm: int) -> int:
        """use hash table and list to record the first pos of circle section"""
        dct = dict()
        # example is x = (x + m) % n
        lst = []
        while x not in dct:
            dct[x] = len(lst)
            lst.append(x)
            x = (x + m) % n

        length = len(lst)
        # the first pos of circle section
        ind = dct[x]
        # current lst is enough
        if tm < length:
            return lst[tm]

        # compute by circle section
        circle = length - ind
        tm -= length
        j = tm % circle
        return lst[ind + j]

    @staticmethod
    def circle_section_pre(n, grid, c, sta, cur, h):
        """circle section with prefix sum"""
        dct = dict()
        lst = []
        cnt = []
        while sta not in dct:
            dct[sta] = len(dct)
            lst.append(sta)
            cnt.append(c)
            sta = cur
            c = 0
            cur = 0
            for i in range(n):
                num = 1 if sta & (1 << i) else 2
                for j in range(n):
                    if grid[i][j] == "1":
                        c += num
                        cur ^= (num % 2) * (1 << j)

        length = len(lst)
        ind = dct[sta]
        pre = [0] * (length + 1)
        for i in range(length):
            pre[i + 1] = pre[i] + cnt[i]

        ans = 0
        if h < length:
            return ans + pre[h]

        circle = length - ind
        circle_cnt = pre[length] - pre[ind]

        h -= length
        ans += pre[length]

        ans += (h // circle) * circle_cnt

        j = h % circle
        ans += pre[ind + j] - pre[ind]
        return ans
import time
from datetime import datetime, timedelta, date


class DateTime:
    def __init__(self):
        self.leap_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.not_leap_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return

    @staticmethod
    def day_interval(y1, m1, d1, y2, m2, d2):
        """the number of days between two date"""
        # 模板: 两个日期之间的间隔天数
        day1 = datetime(y1, m1, d1)
        day2 = datetime(y2, m2, d2)
        return (day1 - day2).days

    @staticmethod
    def time_to_unix(dt):
        time_array = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        # for example dt = 2016-05-05 20:28:54
        timestamp = time.mktime(time_array)
        return timestamp

    @staticmethod
    def unix_to_time(timestamp):
        time_local = time.localtime(timestamp)
        # for example timestamp = 1462451334
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        return dt

    def is_leap_year(self, yy):
        assert sum(self.leap_month) == 366  # for example 2000
        assert sum(self.not_leap_month) == 365  # for example 2001
        return yy % 400 == 0 or (yy % 4 == 0 and yy % 100 != 0)

    @staticmethod
    def get_n_days(yy, mm, dd, n):
        """the day of after n days from yy-mm-dd"""
        now = datetime(yy, mm, dd, 0, 0, 0, 0)
        delta = timedelta(days=n)
        n_days = now + delta
        return n_days.strftime("%Y-%m-%d")

    @staticmethod
    def is_valid_date(date_str):
        try:
            date.fromisoformat(date_str)
        except ValueError as _:
            return False
        else:
            return True

    def all_palindrome_date(self):
        """brute all the palindrome date from 1000-01-01 to 9999-12-31"""
        ans = []
        for y in range(1000, 10000):
            yy = str(y)
            mm = str(y)[::-1][:2]
            dd = str(y)[::-1][2:]
            if self.is_valid_date(f"{yy}-{mm}-{dd}"):
                ans.append(f"{yy}-{mm}-{dd}")
        return ans

    def unix_minute(self, s):
        """minutes start from 0000-00-00-00:00"""
        lst = s.split("-")
        y, m, d = [int(w) for w in lst[:-1]]
        h, minute = [int(w) for w in lst[-1].split(":")]
        day = d + 365 * y + self.leap_year_count(y)
        if self.is_leap_year(y):
            day += sum(self.leap_month[:m - 1])
        else:
            day += sum(self.not_leap_month[:m - 1])
        res = day * 24 * 60 + h * 60 + minute
        return res

    def unix_day(self, s):
        """days start from 0000-00-00-00:00"""
        lst = s.split("-")
        y, m, d = [int(w) for w in lst[:-1]]
        h, minute = [int(w) for w in lst[-1].split(":")]
        day = d + 365 * y + self.leap_year_count(y)
        if self.is_leap_year(y):
            day += sum(self.leap_month[:m - 1])
        else:
            day += sum(self.not_leap_month[:m - 1])
        res = day * 24 * 60 + h * 60 + minute
        return res // (24 * 60)

    def unix_second(self, s):
        """seconds start from 0000-00-00-00:00"""
        lst = s.split("-")
        y, m, d = [int(w) for w in lst[:-1]]
        h, minute, sec = [int(w) for w in lst[-1].split(":")]
        day = d + 365 * y + self.leap_year_count(y)
        if self.is_leap_year(y):
            day += sum(self.leap_month[:m - 1])
        else:
            day += sum(self.not_leap_month[:m - 1])
        res = (day * 24 * 60 + h * 60 + minute) * 60 + sec
        return res

    @staticmethod
    def leap_year_count(y):
        """leap years count small or equal to y"""
        return 1 + y // 4 - y // 100 + y // 400

    @staticmethod
    def get_start_date(y, m, d, hh, mm, ss, x):
        """the time after any seconds"""
        start_date = datetime(year=y, month=m, day=d, hour=hh, minute=mm, second=ss)
        end_date = start_date + timedelta(seconds=x)
        ans = [end_date.year, end_date.month, end_date.day, end_date.hour, end_date.minute, end_date.second]
        return ans

class PreFixSumMatrix:
    def __init__(self, mat):
        self.mat = mat
        self.m, self.n = len(mat), len(mat[0])
        self.pre = [[0] * (self.n + 1) for _ in range(self.m + 1)]
        for i in range(self.m):
            for j in range(self.n):
                self.pre[i + 1][j + 1] = self.pre[i][j + 1] + self.pre[i + 1][j] - self.pre[i][j] + mat[i][j]
        return

    def query(self, xa: int, ya: int, xb: int, yb: int) -> int:
        """left up corner is (xa, ya) and right down corner is (xb, yb)"""
        assert 0 <= xa <= xb <= self.m - 1
        assert 0 <= ya <= yb <= self.n - 1
        return self.pre[xb + 1][yb + 1] - self.pre[xb + 1][ya] - self.pre[xa][yb + 1] + self.pre[xa][ya]


class PreFixSumCube:
    def __init__(self, mat):
        self.mat = mat
        self.n = len(mat)
        self.m = len(mat[0])
        self.p = len(mat[0][0])

        self.prefix_sum = [[[0] * (self.p + 1) for _ in range(self.m + 1)] for _ in range(self.n + 1)]

        for i in range(1, self.n + 1):
            for j in range(1, self.m + 1):
                for k in range(1, self.p + 1):
                    self.prefix_sum[i][j][k] = (mat[i - 1][j - 1][k - 1]
                                                + self.prefix_sum[i - 1][j][k]
                                                + self.prefix_sum[i][j - 1][k]
                                                + self.prefix_sum[i][j][k - 1]
                                                - self.prefix_sum[i - 1][j - 1][k]
                                                - self.prefix_sum[i - 1][j][k - 1]
                                                - self.prefix_sum[i][j - 1][k - 1]
                                                + self.prefix_sum[i - 1][j - 1][k - 1])
        return

    def query(self, x1, x2, y1, y2, z1, z2) -> int:
        """left up corner is (x1, y1, z1) and right down corner is (x2, y2, z2)"""

        assert 1 <= x1 <= x2 <= self.n and 1 <= y1 <= y2 <= self.m and 1 <= z1 <= z2 <= self.p

        def get_sum(x, y, z):
            return self.prefix_sum[x][y][z]

        result = (get_sum(x2, y2, z2)
                  - get_sum(x1 - 1, y2, z2)
                  - get_sum(x2, y1 - 1, z2)
                  - get_sum(x2, y2, z1 - 1)
                  + get_sum(x1 - 1, y1 - 1, z2)
                  + get_sum(x1 - 1, y2, z1 - 1)
                  + get_sum(x2, y1 - 1, z1 - 1)
                  - get_sum(x1 - 1, y1 - 1, z1 - 1))
        return result


class PreFixXorMatrix:
    def __init__(self, mat):
        self.mat = mat
        self.m, self.n = len(mat), len(mat[0])
        self.pre = [[0] * (self.n + 1) for _ in range(self.m + 1)]
        for i in range(self.m):
            for j in range(self.n):
                self.pre[i + 1][j + 1] = self.pre[i][j + 1] ^ self.pre[i + 1][j] ^ self.pre[i][j] ^ mat[i][j]
        return

    def query(self, xa: int, ya: int, xb: int, yb: int) -> int:
        """left up corner is (xa, ya) and right down corner is (xb, yb)"""
        assert 0 <= xa <= xb <= self.m - 1
        assert 0 <= ya <= yb <= self.n - 1
        return self.pre[xb + 1][yb + 1] ^ self.pre[xb + 1][ya] ^ self.pre[xa][yb + 1] ^ self.pre[xa][ya]


class DiffArray:
    def __init__(self):
        return

    @staticmethod
    def get_diff_array(n: int, shifts):
        diff = [0] * n
        for i, j, d in shifts:
            if j + 1 < n:
                diff[j + 1] -= d
            diff[i] += d
        for i in range(1, n):
            diff[i] += diff[i - 1]
        return diff

    @staticmethod
    def get_array_prefix_sum(n: int, lst):
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + lst[i]
        return pre

    @staticmethod
    def get_array_range_sum(pre, left: int, right: int) -> int:
        return pre[right + 1] - pre[left]


class DiffMatrix:
    def __init__(self):
        return

    @staticmethod
    def get_diff_matrix(m: int, n: int, shifts):
        """two dimensional differential array"""
        diff = [[0] * (n + 2) for _ in range(m + 2)]
        # left up corner is (xa, ya) and right down corner is (xb, yb)
        for xa, xb, ya, yb, d in shifts:
            assert 1 <= xa <= xb <= m
            assert 1 <= ya <= yb <= n
            diff[xa][ya] += d
            diff[xa][yb + 1] -= d
            diff[xb + 1][ya] -= d
            diff[xb + 1][yb + 1] += d

        for i in range(1, m + 2):
            for j in range(1, n + 2):
                diff[i][j] += diff[i - 1][j] + diff[i][j - 1] - diff[i - 1][j - 1]

        for i in range(1, m + 1):
            diff[i] = diff[i][1:n + 1]
        return diff[1: m + 1]

    @staticmethod
    def get_diff_matrix2(m, n, shifts):
        """left up corner is (xa, ya) and right down corner is (xb, yb)"""
        diff = [[0] * (n + 1) for _ in range(m + 1)]
        for xa, xb, ya, yb, d in shifts:
            assert 0 <= xa <= xb <= m - 1
            assert 0 <= ya <= yb <= n - 1
            diff[xa][ya] += d
            diff[xa][yb + 1] -= d
            diff[xb + 1][ya] -= d
            diff[xb + 1][yb + 1] += d

        res = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                res[i + 1][j + 1] = res[i + 1][j] + res[i][j + 1] - res[i][j] + diff[i][j]
        return [item[1:] for item in res[1:]]

    @staticmethod
    def get_diff_matrix3(m, n, shifts):
        """left up corner is (xa, ya) and right down corner is (xb, yb)"""
        diff = [0] * ((m + 1) * (n + 1))
        for xa, xb, ya, yb, d in shifts:
            assert 0 <= xa <= xb <= m - 1
            assert 0 <= ya <= yb <= n - 1
            diff[xa * (n + 1) + ya] += d
            diff[xa * (n + 1) + yb + 1] -= d
            diff[(xb + 1) * (n + 1) + ya] -= d
            diff[(xb + 1) * (n + 1) + yb + 1] += d

        res = [0] * ((m + 1) * (n + 1))
        for i in range(m):
            for j in range(n):
                res[(i + 1) * (n + 1) + j + 1] = res[(i + 1) * (n + 1) + j] + res[i * (n + 1) + j + 1] - res[i * (n + 1) + j] + diff[i * (n + 1) + j]
        return [res[i * (n + 1) + 1:(i + 1) * (n + 1)] for i in range(1, m+1)]
import random


class HashMap:
    def __init__(self):
        return

    def gen_result(self):
        return



class Implemention:
    def __init__(self):
        return

    @staticmethod
    def matrix_rotate(matrix):

        """rotate matrix 90 degrees clockwise"""
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                a, b, c, d = matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1], matrix[i][j]
                matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] = a, b, c, d

        """rotate matrix 90 degrees counterclockwise"""
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                a, b, c, d = matrix[j][n - i - 1], matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1]
                matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] = a, b, c, d

        return matrix


class SpiralMatrix:
    def __init__(self):
        return

    @staticmethod
    def joseph_circle(n, m):
        """the last rest for remove the m-th every time in [0,1,...,n-1]"""
        f = 0
        for x in range(2, n + 1):
            f = (m + f) % x
        return f

    @staticmethod
    def num_to_loc(m, n, num):
        """matrix pos from num to loc"""
        # 0123
        # 4567
        m += 1
        return [num // n, num % n]

    @staticmethod
    def loc_to_num(r, c, m, n):
        """matrix pos from loc to num"""
        c += m
        return r * n + n

    @staticmethod
    def get_spiral_matrix_num1(m, n, r, c) -> int:
        """clockwise spiral num at pos [r, c] start from 1"""
        assert 1 <= r <= m and 1 <= c <= n
        num = 1
        while r not in [1, m] and c not in [1, n]:
            num += 2 * m + 2 * n - 4
            r -= 1
            c -= 1
            n -= 2
            m -= 2

        # time complexity is O(m+n)
        x = y = 1
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        d = 0
        while [x, y] != [r, c]:
            a, b = directions[d]
            if not (1 <= x + a <= m and 1 <= y + b <= n):
                d += 1
                a, b = directions[d]
            x += a
            y += b
            num += 1
        return num

    @staticmethod
    def get_spiral_matrix_num2(m, n, r, c) -> int:

        """clockwise spiral num at pos [r, c] start from 1"""
        assert 1 <= r <= m and 1 <= c <= n

        rem = min(r - 1, m - r, c - 1, n - c)
        num = 2 * rem * (m - rem + 1) + 2 * rem * (n - rem + 1) - 4 * rem
        m -= 2 * rem
        n -= 2 * rem
        r -= rem
        c -= rem

        # time complexity is O(1)
        if r == 1:
            num += c
        elif 1 < r <= m and c == n:
            num += n + (r - 1)
        elif r == m and 1 <= c <= n - 1:
            num += n + (m - 1) + (n - c)
        else:
            num += n + (m - 1) + (n - 1) + (m - r)
        return num

    @staticmethod
    def get_spiral_matrix_loc(m, n, num):
        """clockwise spiral pos of num start from 1"""
        assert 1 <= num <= m * n

        def check(x):
            res = 2 * x * (m - x + 1) + 2 * x * (n - x + 1) - 4 * x
            return res < num

        low = 0
        high = max(m // 2, n // 2)
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        rem = high if check(high) else low

        num -= 2 * rem * (m - rem + 1) + 2 * rem * (n - rem + 1) - 4 * rem
        assert num > 0
        m -= 2 * rem
        n -= 2 * rem
        r = c = rem

        if num <= n:
            a = 1
            b = num
        elif n < num <= n + m - 1:
            a = num - n + 1
            b = n
        elif n + (m - 1) < num <= n + (m - 1) + (n - 1):
            a = m
            b = n - (num - n - (m - 1))
        else:
            a = m - (num - n - (n - 1) - (m - 1))
            b = 1
        return [r + a, c + b]
from functools import reduce
from operator import mul


class MdVector:
    def __init__(self, dimension, initial):
        self.dimension = dimension
        self.dp = [initial] * reduce(mul, dimension)
        self.m = len(dimension)
        self.pos = []
        for i in range(self.m):
            self.pos.append(reduce(mul, dimension[i + 1:] + [1]))
        return

    def get(self, lst):
        return sum(x * y for x, y in zip(lst, self.pos))
class Performance:
    def __init__(self):
        return

    def gen_result(self):
        return




class Range:
    def __init__(self):
        return

    @staticmethod
    def range_merge_to_disjoint(lst):
        """range_merge_to_disjoint intervals into disjoint intervals"""
        lst.sort(key=lambda it: it[0])
        ans = []
        x, y = lst[0]
        for a, b in lst[1:]:
            if a <= y:  # [1, 3] + [3, 4] = [1, 4]
                # if wanted range_merge_to_disjoint like [1, 2] + [3, 4] = [1, 4] can change to a <= y+1 or a < y
                y = y if y > b else b
            else:
                ans.append([x, y])
                x, y = a, b
        ans.append([x, y])
        return ans

    @staticmethod
    def minimum_range_cover(s, t, lst, inter=True):
        """calculate the minimum number of intervals in lst for coverage [s, t]"""
        if not lst:
            return -1
        # [1, 3] + [3, 4] = [1, 4] by set inter=True
        # [1, 2] + [3, 4] = [1, 4] by set inter=False
        lst.sort(key=lambda x: [x[0], -x[1]])
        if lst[0][0] != s:
            return -1
        if lst[0][1] >= t:
            return 1
        ans = 1
        end = lst[0][1]
        cur = -1
        for a, b in lst[1:]:
            if end >= t:
                return ans
            # can be next disjoint set
            if (end >= a and inter) or (not inter and end >= a - 1):
                cur = cur if cur > b else b
            else:
                if cur <= end:
                    return -1
                # add new farthest range
                ans += 1
                end = cur
                cur = -1
                if end >= t:
                    return ans
                if (end >= a and inter) or (not inter and end >= a - 1):
                    cur = cur if cur > b else b
                else:
                    return -1  # which is impossible to coverage [s, t]
        if cur >= t:
            ans += 1
            return ans
        return -1

    @staticmethod
    def minimum_interval_coverage(clips, time: int, inter=True) -> int:
        """calculate the minimum number of intervals in clips for coverage [0, time]"""
        assert inter
        assert time >= 0
        if not clips:
            return -1
        if time == 0:
            if min(x for x, _ in clips) > 0:
                return -1
            return 1

        if inter:
            # inter=True is necessary
            post = [0] * time
            for a, b in clips:
                if a < time:
                    post[a] = post[a] if post[a] > b else b
            if not post[0]:
                return -1

            ans = right = pre_end = 0
            for i in range(time):
                right = right if right > post[i] else post[i]
                if i == right:
                    return -1
                if i == pre_end:
                    ans += 1
                    pre_end = right
        else:
            ans = -1
        return ans

    @staticmethod
    def maximum_disjoint_range(lst):
        """select the maximum disjoint intervals"""
        lst.sort(key=lambda x: x[1])
        ans = 0
        end = -math.inf
        for a, b in lst:
            if a >= end:
                ans += 1
                end = b
        return ans

    @staticmethod
    def minimum_point_cover_range(lst):
        """find the minimum number of point such that every range in lst has at least one point"""
        if not lst:
            return 0
        lst.sort(key=lambda it: it[1])
        ans = 1
        a, b = lst[0]
        for c, d in lst[1:]:
            if b < c:
                ans += 1
                b = d
        return ans
from collections import deque
from typing import Optional

from src.basis.tree_node.template import TreeNode


class CodecBFS:

    @staticmethod
    def serialize(root: Optional[TreeNode]) -> str:
        """Encodes a tree to a single strings.
        """
        stack = deque([root]) if root else deque()
        res = []
        while stack:
            node = stack.popleft()
            if not node:
                res.append("n")
                continue
            else:
                res.append(str(node.val))
                stack.append(node.left)
                stack.append(node.right)
        return ",".join(res)

    @staticmethod
    def deserialize(data: str) -> Optional[TreeNode]:
        """Decodes your encoded data to tree.
        """
        if not data:
            return
        lst = deque(data.split(","))
        ans = TreeNode(int(lst.popleft()))
        stack = deque([ans])
        while lst:
            left, right = lst.popleft(), lst.popleft()
            pre = stack.popleft()
            if left != "n":
                pre.left = TreeNode(int(left))
                stack.append(pre.left)
            if right != "n":
                pre.right = TreeNode(int(right))
                stack.append(pre.right)
        return ans


class CodecDFS:

    @staticmethod
    def serialize(root: TreeNode) -> str:
        """Encodes a tree to a single strings.
        """
        def dfs(node):
            if not node:
                return "n"
            return dfs(node.right) + "," + dfs(node.left) + "," + str(node.val)

        return dfs(root)

    @staticmethod
    def deserialize(data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        """
        lst = data.split(",")

        def dfs():
            if not lst:
                return
            val = lst.pop()
            if val == "n":
                return
            root = TreeNode(int(val))
            root.left = dfs()
            root.right = dfs()
            return root

        return dfs()
class MaxStack:
    def __init__(self):
        return

    def gen_result(self):
        return


class MinStack:
    def __init__(self):
        return

    def gen_result(self):
        return
import math
from decimal import Decimal


class TernarySearch:
    """
    the platform of the function must be on the ceil or floor, others can not use ternary search
    """

    def __init__(self):
        return

    @staticmethod
    def find_ceil_point_float(fun, left, right, error=1e-9, high_precision=False):
        """the float point at which the upper convex function obtains its maximum value"""
        while left < right - error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 > dist2:
                right = mid2
            elif dist1 < dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_ceil_point_int(fun, left, right, error=1):
        """the int point at which the upper convex function obtains its maximum value"""
        while left < right - error:
            diff = (right - left) // 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 > dist2:
                right = mid2
            elif dist1 < dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_floor_point_float(fun, left, right, error=1e-9, high_precision=False):
        """The float point when solving the convex function to obtain the minimum value"""
        while left < right - error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 < dist2:
                right = mid2
            elif dist1 > dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_floor_point_int(fun, left, right, error=1, high_precision=False):
        """The int point when solving the convex function to obtain the minimum value"""
        while left < right - error:
            diff = Decimal(right - left) // 3 if high_precision else (right - left) // 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 < dist2:
                right = mid2
            elif dist1 > dist2:
                left = mid1
            else:
                left = mid1
                right = mid2
        return left

    @staticmethod
    def find_ceil_value_float(fun, left, right, error=1e-9, high_precision=False):
        f1, f2 = fun(left), fun(right)
        while abs(f1 - f2) > error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 > dist2:
                right = mid2
                f2 = dist2
            elif dist1 < dist2:
                left = mid1
                f1 = dist1
            else:
                left = mid1
                right = mid2
                f1, f2 = dist1, dist2
        return (f1 + f2) / 2

    @staticmethod
    def find_floor_value_float(fun, left, right, error=1e-9, high_precision=False):
        f1, f2 = fun(left), fun(right)
        while abs(f1 - f2) > error:
            diff = Decimal(right - left) / 3 if high_precision else (right - left) / 3
            mid1 = left + diff
            mid2 = left + 2 * diff
            dist1 = fun(mid1)
            dist2 = fun(mid2)
            if dist1 < dist2:
                right = mid2
                f2 = dist2
            elif dist1 > dist2:
                left = mid1
                f1 = dist1
            else:
                left = mid1
                right = mid2
                f1, f2 = dist1, dist2
        return (f1 + f2) / 2


class TriPartPackTriPart:
    def __init__(self):
        return

    @staticmethod
    def find_floor_point_float(target, left_x, right_x, low_y, high_y):
        # Find the smallest coordinate [x, y] to minimize the target of the objective function
        error = 5e-8

        def optimize(y):
            # The loss function
            low_ = left_x
            high_ = right_x
            while low_ < high_ - error:
                diff_ = (high_ - low_) / 3
                mid1_ = low_ + diff_
                mid2_ = low_ + 2 * diff_
                dist1_ = target(mid1_, y)
                dist2_ = target(mid2_, y)
                if dist1_ < dist2_:
                    high_ = mid2_
                elif dist1_ > dist2_:
                    low_ = mid1_
                else:
                    low_ = mid1_
                    high_ = mid2_
            return low_, target(low_, y)

        low = low_y
        high = high_y
        while low < high - error:
            diff = (high - low) / 3
            mid1 = low + diff
            mid2 = low + 2 * diff
            _, dist1 = optimize(mid1)
            _, dist2 = optimize(mid2)
            if dist1 < dist2:
                high = mid2
            elif dist1 > dist2:
                low = mid1
            else:
                low = mid1
                high = mid2
        res_x, r = optimize(low)
        res_y = low
        return [res_x, res_y, math.sqrt(r)]
from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeOrder:
    def __init__(self):
        return

    @staticmethod
    def post_order(root: Optional[TreeNode]) -> List[int]:
        ans = []
        stack = [[root, 1]] if root else []
        while stack:
            node, state = stack.pop()
            if state:
                stack.append([node, 0])
                if node.right:
                    stack.append([node.right, 1])
                if node.left:
                    stack.append([node.left, 1])
            else:
                ans.append(node.val)
        return ans

    @staticmethod
    def pre_order(root: Optional[TreeNode]) -> List[int]:
        ans = []
        stack = [[root, 1]] if root else []
        while stack:
            node, state = stack.pop()
            if state:
                if node.right:
                    stack.append([node.right, 1])
                if node.left:
                    stack.append([node.left, 1])
                stack.append([node, 0])
            else:
                ans.append(node.val)
        return ans

    @staticmethod
    def in_order(root: Optional[TreeNode]) -> List[int]:
        ans = []
        stack = [[root, 1]] if root else []
        while stack:
            node, state = stack.pop()
            if state:
                if node.right:
                    stack.append([node.right, 1])
                stack.append([node, 0])
                if node.left:
                    stack.append([node.left, 1])
            else:
                ans.append(node.val)
        return ans


class TwoPointer:
    def __init__(self):
        return

    @staticmethod
    def window(nums):
        n = len(nums)
        ans = j = 0
        dct = dict()
        for i in range(n):
            while j < n and (nums[j] in dct or not dct
                             or (abs(max(dct) - nums[j]) <= 2 and abs(min(dct) - nums[j]) <= 2)):
                dct[nums[j]] = dct.get(nums[j], 0) + 1
                j += 1
            ans += j - i
            dct[nums[i]] -= 1
            if not dct[nums[i]]:
                del dct[nums[i]]
        return ans

    @staticmethod
    def circle_array(arr):
        """circular array pointer movement"""
        n = len(arr)
        ans = 0
        for i in range(n):
            ans = max(ans, arr[i] + arr[(i + n - 1) % n])
        return ans

    @staticmethod
    def fast_and_slow(head):
        """fast and slow pointers to determine whether there are rings in the linked list"""
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    @staticmethod
    def same_direction(nums):
        """two pointers in the same direction to find the longest subsequence without repeating elements"""
        n = len(nums)
        ans = j = 0
        pre = set()
        for i in range(n):
            while j < n and nums[j] not in pre:
                pre.add(nums[j])
                j += 1
            ans = ans if ans > j - i else j - i
            pre.discard(nums[i])
        return ans

    @staticmethod
    def opposite_direction(nums, target):
        """two pointers in the opposite direction to find two numbers equal target in ascending array"""
        n = len(nums)
        i, j = 0, n - 1
        while i < j:
            cur = nums[i] + nums[j]
            if cur > target:
                j -= 1
            elif cur < target:
                i += 1
            else:
                return True
        return False


class SlidingWindowAggregation:
    """SlidingWindowAggregation

    Api:
    1. append value to tail,O(1).
    2. pop value from head,O(1).
    3. query aggregated value in window,O(1).
    """

    def __init__(self, e, op):
        # Sliding Window Maintenance and Query Aggregated Information
        """
        Args:
            e: unit element
            op: range_merge_to_disjoint function
        """
        self.stack0 = []
        self.agg0 = []
        self.stack2 = []
        self.stack3 = []
        self.e = e
        self.e0 = self.e
        self.e1 = self.e
        self.size = 0
        self.op = op

    def append(self, value) -> None:
        if not self.stack0:
            self.push0(value)
            self.transfer()
        else:
            self.push1(value)
        self.size += 1

    def popleft(self) -> None:
        if not self.size:
            return
        if not self.stack0:
            self.transfer()
        self.stack0.pop()
        self.stack2.pop()
        self.e0 = self.stack2[-1] if self.stack2 else self.e
        self.size -= 1

    def query(self):
        return self.op(self.e0, self.e1)

    def push0(self, value):
        self.stack0.append(value)
        self.e0 = self.op(value, self.e0)
        self.stack2.append(self.e0)

    def push1(self, value):
        self.agg0.append(value)
        self.e1 = self.op(self.e1, value)
        self.stack3.append(self.e1)

    def transfer(self):
        while self.agg0:
            self.push0(self.agg0.pop())
        while self.stack3:
            self.stack3.pop()
        self.e1 = self.e

    def __len__(self):
        return self.size
import random
from collections import Counter
from functools import cmp_to_key


class VariousSort:
    def __init__(self):
        return

    @staticmethod
    def insertion_sort(nums):
        n = len(nums)
        for i in range(1, n):
            key = nums[i]
            j = i - 1
            while j >= 0 and nums[j] > key:
                nums[j + 1] = nums[j]
                j = j - 1
            nums[j + 1] = key
        return nums

    @staticmethod
    def counting_sort(nums):
        count = Counter(nums)
        keys = sorted(count.keys())
        rank = 0
        for key in keys:
            while count[key]:
                nums[rank] = key
                count[key] -= 1
                rank += 1
        return nums

    @staticmethod
    def quick_sort_two(lst):
        n = len(lst)

        def quick_sort(i, j):
            if i >= j:
                return

            # First find the smaller divide and conquer sort
            val = lst[random.randint(i, j)]
            left = i
            for k in range(i, j + 1):
                if lst[k] < val:
                    lst[k], lst[left] = lst[left], lst[k]
                    left += 1
            quick_sort(i, left - 1)

            # Then find the larger divide and conquer sort on the right
            for k in range(i, j + 1):
                if lst[k] == val:
                    lst[k], lst[left] = lst[left], lst[k]
                    left += 1
            quick_sort(left, j)
            return

        quick_sort(0, n - 1)
        return lst

    def range_merge_to_disjoint_sort(self, nums):

        if len(nums) > 1:
            mid = len(nums) // 2
            left = nums[:mid]
            right = nums[mid:]

            self.range_merge_to_disjoint_sort(left)
            self.range_merge_to_disjoint_sort(right)

            # Merge ordered lists using pointers
            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    nums[k] = left[i]
                    i += 1
                else:
                    nums[k] = right[j]
                    j += 1
                k += 1

            while i < len(left):
                nums[k] = left[i]
                i += 1
                k += 1

            while j < len(right):
                nums[k] = right[j]
                j += 1
                k += 1
        return nums

    @staticmethod
    def range_merge_to_disjoint_sort_inverse_pair(nums):
        """Use range_merge_to_disjoint sort to calculate the minimum number of times needed
        to make an array sorted by exchanging only adjacent elements
        which is equal the number of reverse_order_pair
        """

        ans = 0
        n = len(nums)
        arr = [0] * n
        stack = [(0, n - 1)]
        while stack:
            left, right = stack.pop()
            if left >= 0:
                if left >= right:
                    continue
                mid = (left + right) // 2
                stack.append((~left, right))
                stack.append((left, mid))
                stack.append((mid + 1, right))
            else:
                left = ~left
                mid = (left + right) // 2
                i, j = left, mid + 1
                k = left
                while i <= mid and j <= right:
                    if nums[i] <= nums[j]:
                        arr[k] = nums[i]
                        i += 1
                    else:
                        arr[k] = nums[j]
                        j += 1
                        ans += mid - i + 1
                    k += 1
                while i <= mid:
                    arr[k] = nums[i]
                    i += 1
                    k += 1
                while j <= right:
                    arr[k] = nums[j]
                    j += 1
                    k += 1

                for i in range(left, right + 1):
                    nums[i] = arr[i]
        return ans

    @staticmethod
    def heap_sort(nums):

        def sift_down(start, end):

            parent = int(start)
            child = int(parent * 2 + 1)

            while child <= end:

                if child + 1 <= end and nums[child] < nums[child + 1]:
                    child += 1

                if nums[parent] >= nums[child]:
                    return

                else:
                    nums[parent], nums[child] = nums[child], nums[parent]
                    parent = child
                    child = int(parent * 2 + 1)
            return

        length = len(nums)

        i = (length - 1 - 1) / 2
        while i >= 0:
            sift_down(i, length - 1)
            i -= 1

        i = length - 1
        while i > 0:
            nums[0], nums[i] = nums[i], nums[0]
            sift_down(0, i - 1)
            i -= 1
        return nums

    @staticmethod
    def shell_sort(nums):
        length = len(nums)
        h = 1
        while h < length / 3:
            h = int(3 * h + 1)
        while h >= 1:
            for i in range(h, length):
                j = i
                while j >= h and nums[j] < nums[j - h]:
                    nums[j], nums[j - h] = nums[j - h], nums[j]
                    j -= h
            h = int(h / 3)
        return nums

    @staticmethod
    def bucket_sort(nums):
        min_num = min(nums)
        max_num = max(nums)
        bucket_range = (max_num - min_num) / len(nums)
        count_list = [[] for _ in range(len(nums) + 1)]
        for i in nums:
            count_list[int((i - min_num) // bucket_range)].append(i)
        nums.clear()
        for i in count_list:
            for j in sorted(i):
                nums.append(j)
        return nums

    @staticmethod
    def bubble_sort(nums):
        n = len(nums)
        flag = True
        while flag:
            flag = False
            for i in range(n - 1):
                if nums[i] > nums[i + 1]:
                    flag = True
                    nums[i], nums[i + 1] = nums[i + 1], nums[i]
        return nums

    @staticmethod
    def selection_sort(nums):
        n = len(nums)
        for i in range(n):
            ith = i
            for j in range(i + 1, n):
                if nums[j] < nums[ith]:
                    ith = j
            nums[i], nums[ith] = nums[ith], nums[i]
        return nums

    @staticmethod
    def defined_sort(nums):

        def compare(a, b):
            if a < b:
                return -1
            elif a > b:
                return 1
            return 0

        def compare2(x, y):
            a = int(x + y)
            b = int(y + x)
            if a < b:
                return -1
            elif a > b:
                return 1
            return 0

        def compare3(x, y):
            a = x + y
            b = y + x
            if a < b:
                return -1
            elif a > b:
                return 1
            return 0

        nums.sort(key=cmp_to_key(compare3))
        nums.sort(key=cmp_to_key(compare2))
        nums.sort(key=cmp_to_key(compare))
        return nums

    @staticmethod
    def minimum_money(transactions) -> int:

        def check(ls):
            x, y = ls[0], ls[1]
            res = [0, 0]
            if x > y:
                res[0] = 0
                res[1] = y
            else:
                res[0] = 1
                res[1] = -x
            return res

        transactions.sort(key=lambda it: check(it))
        ans = cur = 0
        for a, b in transactions:
            if cur < a:
                ans += a - cur
                cur = a
            cur += b - a
        return ans
import random
from collections import Counter


class HashWithRandomSeedEscapeExplode:
    def __int__(self):
        return

    @staticmethod
    def get_cnt(nums):
        """template of associative array"""
        seed = random.randint(0, 10 ** 9 + 7)
        return Counter([num ^ seed for num in nums])
class SegBitSet:
    def __init__(self, n):
        self.n = n
        self.val = 0
        return

    def update(self, ll, rr):
        assert 0 <= ll <= rr <= self.n - 1
        mask = ((1 << (rr - ll + 1)) - 1) << ll
        self.val ^= mask
        return

    def query(self, ll, rr):
        assert 0 <= ll <= rr <= self.n - 1
        if ll == 0 and rr == self.n - 1:
            return self.val.bit_count()
        mask = ((1 << (rr - ll + 1)) - 1) << ll
        return (self.val & mask).bit_count()
from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class ListNodeOperation:
    def __init__(self):
        return

    @staticmethod
    def node_to_lst(node: ListNode) -> List[int]:
        lst = []
        while node:
            lst.append(node.val)
            node = node.next
        return lst

    @staticmethod
    def lst_to_node(lst: List[int]) -> ListNode:
        node = ListNode(-1)
        pre = node
        for num in lst:
            pre.next = ListNode(num)
            pre = pre.next
        return node.next

    @staticmethod
    def node_to_num(node: ListNode) -> int:
        num = 0
        while node:
            num = num * 10 + node.val
            node = node.next
        return num

    @staticmethod
    def num_to_node(num: int) -> ListNode:
        node = ListNode(-1)
        pre = node
        for x in str(num):
            pre.next = ListNode(int(x))
            pre = pre.next
        return node.next
from collections import deque


class PriorityQueue:
    def __init__(self):
        return

    @staticmethod
    def sliding_window(nums, k: int, method="max"):
        assert k >= 1
        if method == "min":
            nums = [-num for num in nums]
        n = len(nums)
        stack = deque()
        ans = []
        for i in range(n):
            while stack and stack[0][1] <= i - k:
                stack.popleft()
            while stack and stack[-1][0] <= nums[i]:
                stack.pop()
            stack.append([nums[i], i])
            if i >= k - 1:
                ans.append(stack[0][0])
        if method == "min":
            ans = [-num for num in ans]
        return ans

    @staticmethod
    def sliding_window_all(nums, k: int, method="max"):
        assert k >= 1
        if method == "min":
            nums = [-num for num in nums]
        n = len(nums)
        stack = deque()
        ans = []
        for i in range(n):
            while stack and stack[0][1] <= i - k:
                stack.popleft()
            while stack and stack[-1][0] <= nums[i]:
                stack.pop()
            stack.append([nums[i], i])
            ans.append(stack[0][0])
        if method == "min":
            ans = [-num for num in ans]
        return ans
import heapq


class QuickMonotonicStack:
    def __init__(self):
        return

    @staticmethod
    def pipline_general(nums):
        """template of index as pre bound and post bound in monotonic stack"""
        n = len(nums)
        post = [n - 1] * n  # initial can be n or n-1 or -1 dependent on usage
        pre = [0] * n  # initial can be 0 or -1 dependent on usage
        stack = []
        for i in range(n):  # can be also range(n-1, -1, -1) dependent on usage
            while stack and nums[stack[-1]] < nums[i]:  # can be < or > or <=  or >=  dependent on usage
                post[stack.pop()] = i - 1  # can be i or i-1 dependent on usage
            if stack:  # which can be done only pre and post are no-repeat such as post bigger and pre not-bigger
                pre[i] = stack[-1] + 1  # can be stack[-1] or stack[-1]-1 dependent on usage
            stack.append(i)

        # strictly smaller at pre or post
        post_min = [n - 1] * n
        pre_min = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[i] < nums[stack[-1]]:
                post_min[stack.pop()] = i - 1
            stack.append(i)
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[i] < nums[stack[-1]]:
                pre_min[stack.pop()] = i + 1
            stack.append(i)

        # strictly bigger at pre or post
        post_max = [n - 1] * n
        pre_max = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                post_max[stack.pop()] = i - 1
            stack.append(i)
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[i] > nums[stack[-1]]:
                pre_max[stack.pop()] = i + 1
            stack.append(i)
        return

    @staticmethod
    def pipline_general_2(nums):
        """template of post second strictly larger or pre second strictly larger
        which can also be solved by offline queries with sorting and binary search
        """
        n = len(nums)
        # next strictly larger elements
        post = [-1] * n
        # next and next strictly larger elements
        post2 = [-1] * n
        stack1 = []
        stack2 = []
        for i in range(n):
            while stack2 and stack2[0][0] < nums[i]:
                post2[heapq.heappop(stack2)[1]] = i
            while stack1 and nums[stack1[-1]] < nums[i]:
                j = stack1.pop()
                post[j] = i
                heapq.heappush(stack2, [nums[j], j])
            stack1.append(i)

        # previous strictly larger elements
        pre = [-1] * n
        # previous and previous strictly larger elements
        pre2 = [-1] * n
        stack1 = []
        stack2 = []
        for i in range(n - 1, -1, -1):
            while stack2 and stack2[0][0] < nums[i]:
                pre2[heapq.heappop(stack2)[1]] = i
            while stack1 and nums[stack1[-1]] < nums[i]:
                j = stack1.pop()
                pre[j] = i
                heapq.heappush(stack2, [nums[j], j])
            stack1.append(i)
        return


class MonotonicStack:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)

        self.pre_bigger = [-1] * self.n
        self.pre_bigger_equal = [-1] * self.n
        self.pre_smaller = [-1] * self.n
        self.pre_smaller_equal = [-1] * self.n

        self.post_bigger = [-1] * self.n
        self.post_bigger_equal = [-1] * self.n
        self.post_smaller = [-1] * self.n
        self.post_smaller_equal = [-1] * self.n

        self.gen_result()
        return

    def gen_result(self):

        stack = []
        for i in range(self.n):
            while stack and self.nums[i] >= self.nums[stack[-1]]:
                self.post_bigger_equal[stack.pop()] = i
            if stack:
                self.pre_bigger[i] = stack[-1]
            stack.append(i)

        stack = []
        for i in range(self.n):
            while stack and self.nums[i] <= self.nums[stack[-1]]:
                self.post_smaller_equal[stack.pop()] = i
            if stack:
                self.pre_smaller[i] = stack[-1]
            stack.append(i)

        stack = []
        for i in range(self.n - 1, -1, -1):
            while stack and self.nums[i] >= self.nums[stack[-1]]:
                self.pre_bigger_equal[stack.pop()] = i
            if stack:
                self.post_bigger[i] = stack[-1]
            stack.append(i)

        stack = []
        for i in range(self.n - 1, -1, -1):
            while stack and self.nums[i] <= self.nums[stack[-1]]:
                self.pre_smaller_equal[stack.pop()] = i
            if stack:
                self.post_smaller[i] = stack[-1]
            stack.append(i)

        return


class Rectangle:
    def __init__(self):
        return

    @staticmethod
    def compute_area(pre):
        """Calculate maximum rectangle area based on height using monotonic stack"""

        m = len(pre)
        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and pre[stack[-1]] > pre[i]:
                right[stack.pop()] = i - 1
            if stack:
                left[i] = stack[-1] + 1
            stack.append(i)

        ans = 0
        for i in range(m):
            cur = pre[i] * (right[i] - left[i] + 1)
            ans = ans if ans > cur else cur
        return ans

    @staticmethod
    def compute_width(pre):
        """Calculate maximum rectangle area based on height using monotonic stack"""

        m = len(pre)
        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and pre[stack[-1]] > pre[i]:
                right[stack.pop()] = i - 1
            if stack:
                left[i] = stack[-1] + 1
            stack.append(i)

        ans = [(left[i], right[i]) for i in range(m)]
        return ans

    @staticmethod
    def compute_number(pre):
        """Use monotonic stack to calculate the number of rectangles based on height"""
        n = len(pre)
        right = [n - 1] * n
        left = [0] * n
        stack = []
        for j in range(n):
            while stack and pre[stack[-1]] > pre[j]:
                right[stack.pop()] = j - 1
            if stack:
                left[j] = stack[-1] + 1
            stack.append(j)

        ans = 0
        for j in range(n):
            ans += (right[j] - j + 1) * (j - left[j] + 1) * pre[j]
        return ans
import heapq
from collections import defaultdict


class HeapqMedian:
    def __init__(self, mid):
        """median maintenance by heapq with odd length array"""
        self.mid = mid
        self.left = []
        self.right = []
        self.left_sum = 0
        self.right_sum = 0
        return

    def add(self, num):

        if num > self.mid:
            heapq.heappush(self.right, num)
            self.right_sum += num
        else:
            heapq.heappush(self.left, -num)
            self.left_sum += num
        n = len(self.left) + len(self.right)

        if n % 2 == 0:
            # maintain equal length
            if len(self.left) > len(self.right):
                self.right_sum += self.mid
                heapq.heappush(self.right, self.mid)
                self.mid = -heapq.heappop(self.left)
                self.left_sum -= self.mid
            elif len(self.right) > len(self.left):
                heapq.heappush(self.left, -self.mid)
                self.left_sum += self.mid
                self.mid = heapq.heappop(self.right)
                self.right_sum -= self.mid
        return

    def query(self):
        return self.mid


class KthLargest:
    def __init__(self, k: int, nums):
        self.heap = [num for num in nums]
        self.k = k
        heapq.heapify(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        while len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]


class FindMedian:
    def __init__(self):
        self.small = []
        self.big = []
        self.big_dct = defaultdict(int)
        self.small_dct = defaultdict(int)
        self.big_cnt = 0
        self.small_cnt = 0

    def delete(self):
        while self.small and not self.small_dct[-self.small[0]]:
            heapq.heappop(self.small)
        while self.big and not self.big_dct[self.big[0]]:
            heapq.heappop(self.big)
        return

    def change(self):
        self.delete()
        while self.small and self.big and -self.small[0] > self.big[0]:
            self.small_dct[-self.small[0]] -= 1
            self.big_dct[-self.small[0]] += 1
            self.small_cnt -= 1
            self.big_cnt += 1
            heapq.heappush(self.big, -heapq.heappop(self.small))
            self.delete()
        return

    def balance(self):
        self.delete()
        while self.small_cnt > self.big_cnt:
            self.small_dct[-self.small[0]] -= 1
            self.big_dct[-self.small[0]] += 1
            heapq.heappush(self.big, -heapq.heappop(self.small))
            self.small_cnt -= 1
            self.big_cnt += 1
            self.delete()

        while self.small_cnt < self.big_cnt - 1:
            self.small_dct[self.big[0]] += 1
            self.big_dct[self.big[0]] -= 1
            heapq.heappush(self.small, -heapq.heappop(self.big))
            self.small_cnt += 1
            self.big_cnt -= 1
            self.delete()
        return


    def add(self, num):
        if not self.big or self.big[0] < num:
            self.big_dct[num] += 1
            heapq.heappush(self.big, num)
            self.big_cnt += 1
        else:
            self.small_dct[num] += 1
            heapq.heappush(self.small, -num)
            self.small_cnt += 1
        self.change()
        self.balance()
        return

    def remove(self, num):
        self.change()
        self.balance()
        if self.big_dct[num]:
            self.big_cnt -= 1
            self.big_dct[num] -= 1
        else:
            self.small_cnt -= 1
            self.small_dct[num] -= 1
        self.change()
        self.balance()
        return

    def find_median(self):
        self.change()
        self.balance()
        if self.big_cnt == self.small_cnt:
            return (-self.small[0] + self.big[0]) // 2
        return self.big[0]


class MedianFinder:
    def __init__(self):
        self.pre = []
        self.post = []

    def add_num(self, num: int) -> None:
        if len(self.pre) != len(self.post):
            heapq.heappush(self.pre, -heapq.heappushpop(self.post, num))
        else:
            heapq.heappush(self.post, -heapq.heappushpop(self.pre, -num))

    def find_median(self) -> float:
        return self.post[0] if len(self.pre) != len(self.post) else (self.post[0] - self.pre[0]) / 2
import math
import random
from collections import defaultdict
from typing import List

from src.data_structure.sorted_list.template import SortedList
from src.data_structure.tree_array.template import PointAddRangeSum



class RangeLongestRegularBrackets:
    def __init__(self, n):
        """query the longest regular brackets of static range"""
        self.n = n
        self.cover = [0] * (4 * n)
        self.left = [0] * (4 * n)
        self.right = [0] * (4 * n)
        return

    def build(self, brackets: str):
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

    def range_longest_regular_brackets(self, left, right):
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
        self.cover = [-math.inf] * (4 * n)
        self.lazy_tag = [-math.inf] * (4 * n)

    def _make_tag(self, i, val):
        self.cover[i] = max(self.cover[i], val)
        self.lazy_tag[i] = max(self.lazy_tag[i], val)
        return

    def _push_up(self, i):
        self.cover[i] = max(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def _push_down(self, i):
        if self.lazy_tag[i] != -math.inf:
            self.cover[i << 1] = max(self.cover[i << 1], self.lazy_tag[i])
            self.cover[(i << 1) | 1] = max(self.cover[(i << 1) | 1], self.lazy_tag[i])
            self.lazy_tag[i << 1] = max(self.lazy_tag[i << 1], self.lazy_tag[i])
            self.lazy_tag[(i << 1) | 1] = max(self.lazy_tag[(i << 1) | 1], self.lazy_tag[i])
            self.lazy_tag[i] = -math.inf
        return

    def build(self, nums):

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

        stack = [(0, self.n - 1, 1)]
        highest = -math.inf
        while stack:
            a, b, i = stack.pop()
            if left <= a and b <= right:
                highest = max(highest, self.cover[i])
                continue
            self._push_down(i)
            m = a + (b - a) // 2
            if left <= m:
                stack.append((a, m, i << 1))
            if right > m:
                stack.append((m + 1, b, (i << 1) | 1))
        return highest

    def range_max_bisect_left(self, left, right, val):
        """binary search with segment tree like"""

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


class RangeAscendRangeMaxIndex:
    def __init__(self, n, initial=math.inf):
        self.n = n
        self.initial = initial
        self.ceil = [-self.initial] * (4 * n)
        self.index = [-self.initial] * (4 * n)
        self.ceil_tag = [-self.initial] * (4 * n)
        self.index_tag = [-self.initial] * (4 * n)

    def _make_tag(self, i, ind, val):
        if val > self.ceil[i]:
            self.ceil[i] = val
            self.index[i] = ind
        if val > self.ceil_tag[i]:
            self.ceil_tag[i] = val
            self.index_tag[i] = ind
        return

    def _push_up(self, i):
        if self.ceil[i << 1] > self.ceil[(i << 1) | 1]:
            self.ceil[i] = self.ceil[i << 1]
            self.index[i] = self.index[i << 1]
        else:
            self.ceil[i] = self.ceil[(i << 1) | 1]
            self.index[i] = self.index[(i << 1) | 1]
        return

    def _push_down(self, i):
        if self.ceil_tag[i] != -self.initial:
            if self.ceil[i << 1] < self.ceil_tag[i]:
                self.ceil[i << 1] = self.ceil_tag[i]
                self.index[i << 1] = self.index_tag[i]

            if self.ceil_tag[i] > self.ceil_tag[i << 1]:
                self.ceil_tag[i << 1] = self.ceil_tag[i]
                self.index_tag[i << 1] = self.index_tag[i]

            if self.ceil[(i << 1) | 1] < self.ceil_tag[i]:
                self.ceil[(i << 1) | 1] = self.ceil_tag[i]
                self.index[(i << 1) | 1] = self.index_tag[i]

            if self.ceil_tag[i] > self.ceil_tag[(i << 1) | 1]:
                self.ceil_tag[(i << 1) | 1] = self.ceil_tag[i]
                self.index_tag[(i << 1) | 1] = self.index_tag[i]

            self.ceil_tag[i] = -self.initial
            self.index_tag[i] = -self.initial
        return

    def build(self, nums):

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, s, nums[s])
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
                nums[s] = self.ceil[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_ascend(self, left, right, ind, val):

        stack = [(0, self.n - 1, 1)]
        while stack:
            a, b, i = stack.pop()
            if i >= 0:
                if left <= a and b <= right:
                    self._make_tag(i, ind, val)
                    continue
                self._push_down(i)
                stack.append((a, b, ~i))
                m = a + (b - a) // 2
                if left <= m:
                    stack.append((a, m, i << 1))
                if right > m:
                    stack.append((m + 1, b, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_max_index(self, left, right):

        stack = [(0, self.n - 1, 1)]
        highest = -self.initial
        ind = 0
        while stack:
            a, b, i = stack.pop()
            if left <= a and b <= right:
                if self.ceil[i] > highest:
                    highest = self.ceil[i]
                    ind = self.index[i]
                continue
            self._push_down(i)
            m = a + (b - a) // 2
            if left <= m:
                stack.append((a, m, i << 1))
            if right > m:
                stack.append((m + 1, b, (i << 1) | 1))
        return highest, ind


class RangeAddRangeAvgDev:
    def __init__(self, n):
        """range_add|range_avg|range_dev"""
        self.n = n
        self.cover = [0] * (4 * self.n)  # x sum of range
        self.cover_square = [0] * (4 * self.n)  # x^2 sum of range
        self.lazy_tag = [0] * (4 * self.n)
        return

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.cover_square[i] = self.cover_square[i << 1] + self.cover_square[(i << 1) | 1]
        return

    def _make_tag(self, i, s, t, val):
        self.cover_square[i] += self.cover[i] * 2 * val + (t - s + 1) * val * val
        self.cover[i] += val * (t - s + 1)
        self.lazy_tag[i] += val
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i]:
            self._make_tag(i << 1, s, m, self.lazy_tag[i])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy_tag[i])
            self.lazy_tag[i] = 0

    def build(self, nums):
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

    def range_add(self, left, right, val):
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

    def range_avg_dev(self, left, right):
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


class RangeAddRangePrePreSum:
    def __init__(self, n, mod=10 ** 9 + 7):
        self.mod = mod
        self.n = n
        self.cover = [0] * (4 * self.n)  # range sum
        self.lazy_tag = [0] * (4 * self.n)  # lazy tag
        self.floor = [0] * (4 * self.n)  # range min
        self.ceil = [0] * (4 * self.n)  # range max
        self.pre_pre = [0] * (4 * self.n)
        self.pre_pre_pre = [0] * (4 * self.n)
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
                m = s + (t - s) // 2
                self._push_up(i, s, m, t)
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i]:
            self.cover[i << 1] += self.lazy_tag[i] * (m - s + 1)
            self.cover[(i << 1) | 1] += self.lazy_tag[i] * (t - m)

            self.floor[i << 1] += self.lazy_tag[i]
            self.floor[(i << 1) | 1] += self.lazy_tag[i]

            self.ceil[i << 1] += self.lazy_tag[i]
            self.ceil[(i << 1) | 1] += self.lazy_tag[i]

            self.lazy_tag[i << 1] += self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] += self.lazy_tag[i]

            self.pre_pre[i] += self.lazy_tag[i] * (1 + t - s + 1) * (t - s + 1)
            self.pre_pre[i] %= self.mod
            self.lazy_tag[i] = 0

    def _push_up(self, i, s, m, t):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.ceil[i] = max(self.ceil[i << 1], self.ceil[(i << 1) | 1])
        self.floor[i] = min(self.floor[i << 1], self.floor[(i << 1) | 1])
        self.pre_pre[i] = self.pre_pre[i << 1] + self.pre_pre[(i << 1) | 1] + self.cover[i << 1] * (t - m)
        self.pre_pre[i] %= self.mod
        return

    def _make_tag(self, i, s, t, val):
        self.cover[i] += val * (t - s + 1)
        self.floor[i] += val
        self.ceil[i] += val
        self.pre_pre[i] += val * (t - s + 1)
        self.lazy_tag[i] += val
        return

    def range_add(self, left, right, val):
        # update the range add

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
                m = s + (t - s) // 2
                self._push_up(i, s, m, t)
        return

    def range_sum(self, left, right):
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

    def range_pre_pre(self, left, right):
        # query the range sum
        stack = [(0, self.n - 1, 1)]
        ans = tot = pre = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.pre_pre[i] + pre * (t - s + 1)
                pre += self.cover[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
            ans %= mod
        return ans

    def point_get(self, ind):
        s, t, i = 0, self.n - 1, 1  #
        while True:
            if s == t == ind:
                ans = self.cover[i]
                break
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return ans

    def range_min(self, left, right):
        # query the range min
        stack = [(0, self.n - 1, 1)]
        lowest = math.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                lowest = min(lowest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return lowest

    def range_min_bisect_left(self, val):
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if self.floor[i << 1] <= val:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return t

    def range_max(self, left, right):
        # query the rang max
        stack = [(0, self.n - 1, 1)]
        highest = -math.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = max(highest, self.ceil[i])
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

    def get_pre_pre(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.pre_pre[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_max_bisect_left(self, left, right, val):
        stack = [(0, self.n - 1, 1)]
        res = -1
        while stack and res == -1:
            s, t, i = stack.pop()
            if s == t:
                if left <= s <= right and self.ceil[i] >= val:
                    res = s
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if right > m and self.ceil[(i << 1) | 1] >= val:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m and self.ceil[i << 1] >= val:
                stack.append((s, m, i << 1))
        return res

    def range_sum_bisect_right_non_zero(self, left):
        if not self.range_sum(0, left):
            return math.inf

        stack = [(0, self.n - 1, 1)]
        res = math.inf
        while stack:
            s, t, i = stack.pop()
            if s == t:
                if s <= left:
                    return s
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if s < left and self.cover[i << 1]:
                stack.append((s, m, i << 1))
            if left > m and self.cover[(i << 1) | 1]:
                stack.append((m + 1, t, (i << 1) | 1))
        return res

    def range_sum_bisect_left_non_zero(self, right):
        if not self.range_sum(right, self.n - 1):
            return math.inf

        stack = [(0, self.n - 1, 1)]
        res = math.inf
        while stack:
            s, t, i = stack.pop()
            if s == t:
                if s >= right:
                    return s
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if t >= right and self.cover[(i << 1) | 1]:
                stack.append((m + 1, t, (i << 1) | 1))
            if right <= m and self.cover[i << 1]:
                stack.append((s, m, i << 1))
        return res


class RangeDescendRangeMin:
    def __init__(self, n):
        self.n = n
        self.cover = [math.inf] * (4 * n)
        self.lazy_tag = [math.inf] * (4 * n)

    def _make_tag(self, i, val):
        self.cover[i] = min(self.cover[i], val)
        self.lazy_tag[i] = min(self.lazy_tag[i], val)
        return

    def _push_up(self, i):
        self.cover[i] = min(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def _push_down(self, i):
        if self.lazy_tag[i] != math.inf:
            self.cover[i << 1] = min(self.cover[i << 1], self.lazy_tag[i])
            self.cover[(i << 1) | 1] = min(self.cover[(i << 1) | 1], self.lazy_tag[i])
            self.lazy_tag[i << 1] = min(self.lazy_tag[i << 1], self.lazy_tag[i])
            self.lazy_tag[(i << 1) | 1] = min(self.lazy_tag[(i << 1) | 1], self.lazy_tag[i])
            self.lazy_tag[i] = math.inf
        return

    def build(self, nums):

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

        stack = [(0, self.n - 1, 1)]
        lowest = math.inf
        while stack:
            a, b, i = stack.pop()
            if left <= a and b <= right:
                lowest = min(lowest, self.cover[i])
                continue
            self._push_down(i)
            m = a + (b - a) // 2
            if left <= m:
                stack.append((a, m, i << 1))
            if right > m:
                stack.append((m + 1, b, (i << 1) | 1))
        return lowest


class PointSetRangeMaxSubSumAlter:
    def __init__(self, n, initial=math.inf):
        self.n = n
        self.initial = initial
        self.cover = [0] * (4 * self.n)
        self.val = [(0,) * 4 for _ in range(4 * self.n)]
        self.map = ["00", "01", "10", "11"]
        self.index = []
        for i in range(4):
            for j in range(4):
                if not (self.map[i][-1] == self.map[j][0] == "1"):
                    self.index.append([i, j, int("0b" + self.map[i][0] + self.map[j][-1], 2)])
        return

    def build(self, nums):

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

    def _make_tag(self, i, val):
        self.cover[i] = val
        self.val[i] = (0, 0, 0, val)
        return

    def _push_up(self, i):
        res1 = self.val[i << 1][:]
        res2 = self.val[(i << 1) | 1]
        self.cover[i] = max(self.cover[i << 1], self.cover[(i << 1) | 1])
        res = [max(res1[j], res2[j]) for j in range(4)]

        for x, y, k in self.index:
            res[k] = max(res[k], res1[x] + res2[y])
            self.cover[i] = max(self.cover[i], res[k])
        self.val[i] = tuple(res)
        return

    def point_set_range_max_sub_sum(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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


class PointSetRangeMaxSubSumAlterSignal:
    def __init__(self, n, initial=math.inf):
        self.n = n
        self.initial = initial
        self.cover = [0] * (4 * self.n)
        self.val = [(0,) * 4 for _ in range(4 * self.n)]
        self.map = ["00", "01", "10", "11"]
        # 0- 1+
        self.index = []
        for i in range(4):
            for j in range(4):
                if self.map[i][-1] != self.map[j][0]:
                    self.index.append([i, j, int("0b" + self.map[i][0] + self.map[j][-1], 2)])
        return

    def build(self, nums):

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

    def _make_tag(self, i, val):
        self.cover[i] = val
        self.val[i] = (-val, -math.inf, -math.inf, val)
        return

    def _push_up(self, i):
        res1 = self.val[i << 1][:]
        res2 = self.val[(i << 1) | 1]
        self.cover[i] = max(self.cover[i << 1], self.cover[(i << 1) | 1])
        res = [max(res1[j], res2[j]) for j in range(4)]

        for x, y, k in self.index:
            res[k] = max(res[k], res1[x] + res2[y])
            self.cover[i] = max(self.cover[i], res[k])
        self.val[i] = tuple(res)
        return

    def point_set_range_max_sub_sum(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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


class RangeAddRangeSumMinMax:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * self.n)  # range sum
        self.lazy_tag = [0] * (4 * self.n)  # lazy tag
        self.floor = [0] * (4 * self.n)  # range min
        self.ceil = [0] * (4 * self.n)  # range max
        return

    def build(self, nums):

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                self.floor[i] = self.cover[i] = self.lazy_tag[i] = self.ceil[i] = 0
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

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i]:
            self.cover[i << 1] += self.lazy_tag[i] * (m - s + 1)
            self.cover[(i << 1) | 1] += self.lazy_tag[i] * (t - m)

            self.floor[i << 1] += self.lazy_tag[i]
            self.floor[(i << 1) | 1] += self.lazy_tag[i]

            self.ceil[i << 1] += self.lazy_tag[i]
            self.ceil[(i << 1) | 1] += self.lazy_tag[i]

            self.lazy_tag[i << 1] += self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] += self.lazy_tag[i]

            self.lazy_tag[i] = 0

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.ceil[i] = max(self.ceil[i << 1], self.ceil[(i << 1) | 1])
        self.floor[i] = min(self.floor[i << 1], self.floor[(i << 1) | 1])
        return

    def _make_tag(self, i, s, t, val):
        self.cover[i] += val * (t - s + 1)
        self.floor[i] += val
        self.ceil[i] += val
        self.lazy_tag[i] += val
        return

    def range_add(self, left, right, val):
        # update the range add

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

    def point_get(self, ind):
        s, t, i = 0, self.n - 1, 1  # 
        while True:
            if s == t == ind:
                ans = self.cover[i]
                break
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return ans

    def range_min(self, left, right):
        # query the range min
        stack = [(0, self.n - 1, 1)]
        lowest = math.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                lowest = min(lowest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return lowest

    def range_min_bisect_left(self, val):
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if self.floor[i << 1] <= val:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return t

    def range_max(self, left, right):
        # query the rang max
        stack = [(0, self.n - 1, 1)]
        highest = -math.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = max(highest, self.ceil[i])
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

    def range_max_bisect_left(self, left, right, val):
        stack = [(0, self.n - 1, 1)]
        res = -1
        while stack and res == -1:
            s, t, i = stack.pop()
            if s == t:
                if left <= s <= right and self.ceil[i] >= val:
                    res = s
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if right > m and self.ceil[(i << 1) | 1] >= val:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m and self.ceil[i << 1] >= val:
                stack.append((s, m, i << 1))
        return res

    def range_max_bisect_right(self, left, right, val):
        stack = [(0, self.n - 1, 1)]
        res = -1
        while stack and res == -1:
            s, t, i = stack.pop()
            if s == t:
                if left <= s <= right and self.ceil[i] >= val:
                    res = s
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m and self.ceil[i << 1] >= val:
                stack.append((s, m, i << 1))
            if right > m and self.ceil[(i << 1) | 1] >= val:
                stack.append((m + 1, t, (i << 1) | 1))
        return res

    def range_sum_bisect_right_non_zero(self, left):
        if not self.range_sum(0, left):
            return math.inf

        stack = [(0, self.n - 1, 1)]
        res = math.inf
        while stack:
            s, t, i = stack.pop()
            if s == t:
                if s <= left:
                    return s
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if s < left and self.cover[i << 1]:
                stack.append((s, m, i << 1))
            if left > m and self.cover[(i << 1) | 1]:
                stack.append((m + 1, t, (i << 1) | 1))
        return res

    def range_sum_bisect_left_non_zero(self, right):
        if not self.range_sum(right, self.n - 1):
            return math.inf

        stack = [(0, self.n - 1, 1)]
        res = math.inf
        while stack:
            s, t, i = stack.pop()
            if s == t:
                if s >= right:
                    return s
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if t >= right and self.cover[(i << 1) | 1]:
                stack.append((m + 1, t, (i << 1) | 1))
            if right <= m and self.cover[i << 1]:
                stack.append((s, m, i << 1))
        return res


class RangeAddRangeMaxGainMinGain:
    def __init__(self, n):
        self.n = n
        self.cover1 = [-math.inf] * (4 * self.n)  # max(post-pre)
        self.cover2 = [math.inf] * (4 * self.n)  # min(post-pre)
        self.lazy_tag = [0] * (4 * self.n)  # lazy tag
        self.floor = [0] * (4 * self.n)  # range min
        self.ceil = [0] * (4 * self.n)  # range max
        return

    def build(self, nums):

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

    def _push_down(self, i):
        if self.lazy_tag[i]:
            self.floor[i << 1] += self.lazy_tag[i]
            self.floor[(i << 1) | 1] += self.lazy_tag[i]

            self.ceil[i << 1] += self.lazy_tag[i]
            self.ceil[(i << 1) | 1] += self.lazy_tag[i]

            self.lazy_tag[i << 1] += self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] += self.lazy_tag[i]

            self.lazy_tag[i] = 0

    def _push_up(self, i):
        self.cover1[i] = max(self.cover1[i << 1], self.cover1[(i << 1) | 1],
                             self.ceil[(i << 1) | 1] - self.floor[i << 1])
        self.cover2[i] = min(self.cover2[i << 1], self.cover2[(i << 1) | 1],
                             self.floor[(i << 1) | 1] - self.ceil[i << 1])

        self.ceil[i] = max(self.ceil[i << 1], self.ceil[(i << 1) | 1])
        self.floor[i] = min(self.floor[i << 1], self.floor[(i << 1) | 1])
        return

    def _make_tag(self, i, val):
        self.floor[i] += val
        self.ceil[i] += val
        self.lazy_tag[i] += val
        return

    def range_add(self, left, right, val):
        # update the range add

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, val)
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

    def range_min(self, left, right):
        # query the range min
        stack = [(0, self.n - 1, 1)]
        lowest = math.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                lowest = min(lowest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return lowest

    def range_max_gain_min_gain(self, left, right):
        # query the rang max
        stack = [(0, self.n - 1, 1)]
        ans = [math.inf, -math.inf, -math.inf, math.inf]  # floor, ceil, max_gain, min_gain
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                floor, ceil, max_gain, min_gain = self.floor[i], self.ceil[i], self.cover1[i], self.cover2[i]
                ans[2] = max(ans[2], max_gain, ceil - ans[0])
                ans[3] = min(ans[3], min_gain, floor - ans[1])
                ans[0] = min(ans[0], floor)
                ans[1] = max(ans[1], ceil)
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.ceil[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums


class RangeAddRangeConSubPalindrome:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * self.n)  # range cover
        self.lazy_tag = [0] * (4 * self.n)  # lazy tag
        self.pref = [[0, -1] for _ in range(4 * self.n)]
        self.suf = [[-1, 0] for _ in range(4 * self.n)]
        return

    def build(self, nums):

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

    def _push_down(self, i):
        if self.lazy_tag[i]:
            self._make_tag(i << 1, self.lazy_tag[i])
            self._make_tag((i << 1) | 1, self.lazy_tag[i])
            self.lazy_tag[i] = 0

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] | self.cover[(i << 1) | 1]
        self.pref[i][0] = self.pref[i << 1][0]
        self.pref[i][1] = self.pref[i << 1][1] if self.pref[i << 1][1] != -1 else self.pref[(i << 1) | 1][0]

        self.suf[i][1] = self.suf[(i << 1) | 1][1]
        self.suf[i][0] = self.suf[(i << 1) | 1][0] if self.suf[(i << 1) | 1][0] != -1 else self.suf[i << 1][1]

        if self.suf[i << 1][1] == self.pref[(i << 1) | 1][0] or self.suf[i << 1][1] == self.pref[(i << 1) | 1][1] or \
                self.suf[i << 1][0] == self.pref[(i << 1) | 1][0]:
            self.cover[i] = 1
        return

    def _make_tag(self, i, val):
        self.pref[i][0] = (self.pref[i][0] + val) % 26
        if self.pref[i][1] != -1:
            self.pref[i][1] = (self.pref[i][1] + val) % 26

        self.suf[i][1] = (self.suf[i][1] + val) % 26
        if self.suf[i][0] != -1:
            self.suf[i][0] = (self.suf[i][0] + val) % 26
        self.lazy_tag[i] += val
        self.lazy_tag[i] %= 26
        return

    def range_add(self, left, right, val):
        # update the range add

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, val)
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

    def range_con_sub_palindrome(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        cur = [-1, -1]
        while stack and not ans:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans |= self.cover[i]
                pref, suf = self.pref[i], self.suf[i]
                if cur[1] != -1 and (cur[1] == pref[0] or cur[1] == pref[1] or cur[0] == pref[0]):
                    ans = 1
                cur[0] = suf[0] if suf[0] != -1 else cur[1]
                cur[1] = suf[1]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans


class RangeAddRangeWeightedSum:
    def __init__(self, n):
        self.n = n
        self.weighted_sum = [0] * (4 * self.n)  # range weighted sum
        self.sum = [0] * (4 * self.n)  # range sum
        self.lazy_tag = [0] * (4 * self.n)  # lazy tag
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
                self._push_up(i, s, t)
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i]:
            self._make_tag(i << 1, s, m, self.lazy_tag[i])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy_tag[i])
            self.lazy_tag[i] = 0

    def _push_up(self, i, s, t):
        m = s + (t - s) // 2
        self.weighted_sum[i] = (self.weighted_sum[i << 1]
                                + self.weighted_sum[(i << 1) | 1] + self.sum[(i << 1) | 1] * (m - s + 1))
        self.sum[i] = self.sum[i << 1] + self.sum[(i << 1) | 1]
        return

    def _make_tag(self, i, s, t, val):
        length = t - s + 1
        self.weighted_sum[i] += val * (length + 1) * length // 2
        self.sum[i] += val * length
        self.lazy_tag[i] += val
        return

    def range_add(self, left, right, val):
        # update the range add

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
                self._push_up(i, s, t)
        return

    def range_weighted_sum(self, left, right):
        # query the range sum
        stack = [(0, self.n - 1, 1)]
        ans = pre = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.weighted_sum[i] + pre * self.sum[i]
                pre += t - s + 1
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.sum[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums


class RangeChminChmaxPointGet:
    def __init__(self, n, low_initial=-math.inf, high_initial=math.inf):
        self.n = n
        self.low_initial = low_initial
        self.high_initial = high_initial
        self.low = [self.low_initial] * (4 * self.n)
        self.high = [self.high_initial] * (4 * self.n)
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if s == t:
                self._make_tag(i, nums[s], nums[s])
            else:
                m = s + (t - s) // 2
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
        return

    def _push_down(self, i):
        if not (self.high[i] == self.high_initial and self.low[i] == self.low_initial):
            self._make_tag(i << 1, self.low[i], self.high[i])
            self._make_tag((i << 1) | 1, self.low[i], self.high[i])
            self.high[i] = self.high_initial
            self.low[i] = self.low_initial
        return

    def _merge_tag(self, low1, high1, low2, high2):
        high2 = min(high2, high1)
        high2 = max(high2, low1)
        low2 = max(low2, low1)
        low2 = min(low2, high1)
        return low2, high2

    def _make_tag(self, i, low, high):
        self.low[i], self.high[i] = self._merge_tag(low, high, self.low[i], self.high[i])
        return

    def range_chmin_chmax(self, left, right, low, high):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                self._make_tag(i, low, high)
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return

    def point_get(self, ind):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                ans = self.low[i]
                break
            m = s + (t - s) // 2
            self._push_down(i)
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return ans

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.low[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums


class PointAddRangeSum1Sum2:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover1 = [initial] * 4 * n
        self.cover2 = [initial] * 4 * n
        return

    def _push_up(self, i):
        self.cover1[i] = self.cover1[i << 1] + self.cover1[(i << 1) | 1]
        self.cover2[i] = self.cover2[i << 1] + self.cover2[(i << 1) | 1]
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover1[i], self.cover2[i] = nums[s][0], nums[s][0] * nums[s][1]
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
        nums = [[0, 0] for _ in range(self.n)]
        while stack:
            s, t, i = stack.pop()
            if s == t:
                val = [self.cover1[i], self.cover2[i] // self.cover1[i]]
                nums[s] = val
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_add(self, ind, val):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                self.cover1[i] += val[0]
                self.cover2[i] += val[1]
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

    def range_sum1(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover1[i]
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def range_sum2(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.cover2[i]
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def range_sum2_bisect_left(self, cnt):
        s, t, i = 0, self.n - 1, 1
        ans = 0
        while s < t:
            m = s + (t - s) // 2
            if self.cover1[i << 1] >= cnt:
                s, t, i = s, m, i << 1
            else:
                cnt -= self.cover1[i << 1]
                ans += self.cover2[i << 1]
                s, t, i = m + 1, t, (i << 1) | 1
        return ans, cnt, t


class PointSetPreMinPostMin:

    def __init__(self, n, initial=math.inf):
        self.n = n
        self.initial = initial
        self.pre = [initial] * (4 * n)
        self.post = [initial] * (4 * n)
        return

    def _make_tag(self, i, val):
        self.pre[i] = val
        self.post[i] = val
        return

    def _push_up(self, i):
        self.pre[i] = min(self.pre[i << 1], self.pre[(i << 1) | 1])
        self.post[i] = min(self.post[i << 1], self.post[(i << 1) | 1])
        return

    def build(self, nums):

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
                val = self.pre[i]
                nums[s] = val
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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
        return

    def pre_min(self, ind):
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        while stack:
            s, t, i = stack.pop()
            if t <= ind:
                ans = min(ans, self.pre[i])
                continue
            m = s + (t - s) // 2
            if ind >= s:
                stack.append((s, m, i << 1))
            if ind >= m + 1:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def post_min(self, ind):
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        while stack:
            s, t, i = stack.pop()
            if s >= ind:
                ans = min(ans, self.post[i])
                continue
            m = s + (t - s) // 2
            if m >= ind:
                stack.append((s, m, i << 1))
            if t >= ind:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def bisect_left_post_min(self, val):
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            if self.post[(i << 1) | 1] >= val:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return t

    def bisect_right_pre_min(self, val):
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            if self.pre[i << 1] >= val:
                s, t, i = m + 1, t, (i << 1) | 1
            else:
                s, t, i = s, m, i << 1
        return t


class PointSetPreMaxPostMin:

    def __init__(self, n, initial=math.inf):
        self.n = n
        self.initial = initial
        self.pre = [-initial] * (4 * n)
        self.post = [initial] * (4 * n)
        return

    def _make_tag(self, i, val):
        self.pre[i] = val
        self.post[i] = val
        return

    def _push_up(self, i):
        self.pre[i] = max(self.pre[i << 1], self.pre[(i << 1) | 1])
        self.post[i] = min(self.post[i << 1], self.post[(i << 1) | 1])
        return

    def build(self, nums):

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
                val = self.pre[i]
                nums[s] = val
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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
        return

    def pre_max(self, ind):
        stack = [(0, self.n - 1, 1)]
        ans = -self.initial
        while stack:
            s, t, i = stack.pop()
            if t <= ind:
                ans = max(ans, self.pre[i])
                continue
            m = s + (t - s) // 2
            if ind >= s:
                stack.append((s, m, i << 1))
            if ind >= m + 1:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def post_min(self, ind):
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        while stack:
            s, t, i = stack.pop()
            if s >= ind:
                ans = min(ans, self.post[i])
                continue
            m = s + (t - s) // 2
            if m >= ind:
                stack.append((s, m, i << 1))
            if t >= ind:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def bisect_left_post_min(self, ind, val):
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            if self.post[(i << 1) | 1] >= val and m >= ind:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return t

    def bisect_right_pre_max(self, ind, val):
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            if self.pre[i << 1] <= val and m + 1 <= ind:
                s, t, i = m + 1, t, (i << 1) | 1
            else:
                s, t, i = s, m, i << 1
        return t


class PointAddRangeSumMod5:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover = [(initial,) * 6 for _ in range(4 * n)]
        return

    @classmethod
    def _merge(cls, tup1, tup2):
        res = [tup1[0] + tup2[0]]
        x = tup1[0]
        for i in range(5):
            res.append(tup1[i + 1] + tup2[(i - x) % 5 + 1])
        return tuple(res)

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = (1, nums[i], 0, 0, 0, 0)
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self.cover[i] = self._merge(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0 for _ in range(self.n)]
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.cover[i][0]
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_add(self, ind, val):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                pre = list(self.cover[i])
                pre[0] += val[0]
                pre[1] += val[1]
                self.cover[i] = tuple(pre)
                break
            m = s + (t - s) // 2
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        while i > 1:
            i //= 2
            self.cover[i] = self._merge(self.cover[i << 1], self.cover[(i << 1) | 1])
        return


class RangeAddPointGet:
    def __init__(self, n):
        self.n = n
        self.lazy_tag = [0] * (4 * self.n)
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if s == t:
                self._make_tag(i, nums[s])
            else:
                m = s + (t - s) // 2
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
        return

    def _push_down(self, i):
        if self.lazy_tag[i]:
            self.lazy_tag[i << 1] += self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] += self.lazy_tag[i]
            self.lazy_tag[i] = 0

    def _make_tag(self, i, val):
        self.lazy_tag[i] += val
        return

    def range_add(self, left, right, val):
        stack = [(0, self.n - 1, 1)]  # 
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                self._make_tag(i, val)
                continue

            m = s + (t - s) // 2
            self._push_down(i)

            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return

    def point_get(self, ind):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                ans = self.lazy_tag[i]
                break
            m = s + (t - s) // 2
            self._push_down(i)
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return ans

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.lazy_tag[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums


class RangeSetPointGet:
    def __init__(self, n, initial=-1):
        self.n = n
        self.initial = initial
        self.lazy_tag = [initial] * (4 * self.n)
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if s == t:
                self._make_tag(i, nums[s])
            else:
                m = s + (t - s) // 2
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
        return

    def _push_down(self, i):
        if self.lazy_tag[i] != self.initial:
            self.lazy_tag[i << 1] = self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] = self.lazy_tag[i]
            self.lazy_tag[i] = self.initial

    def _make_tag(self, i, val):
        self.lazy_tag[i] = val
        return

    def range_set(self, left, right, val):
        stack = [(0, self.n - 1, 1)]  # 
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                self._make_tag(i, val)
                continue

            m = s + (t - s) // 2
            self._push_down(i)

            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return

    def point_get(self, ind):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                ans = self.lazy_tag[i]
                break
            m = s + (t - s) // 2
            self._push_down(i)
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return ans

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.lazy_tag[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums


class RangeAscendPointGet:
    def __init__(self, n):
        self.n = n
        self.lazy_tag = [0] * (4 * self.n)
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if s == t:
                self._make_tag(i, nums[s])
            else:
                m = s + (t - s) // 2
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
        return

    @classmethod
    def _max(cls, x, y):
        return x if x > y else y

    def _push_down(self, i):
        if self.lazy_tag[i]:
            self.lazy_tag[i << 1] = max(self.lazy_tag[i << 1], self.lazy_tag[i])
            self.lazy_tag[(i << 1) | 1] = max(self.lazy_tag[(i << 1) | 1], self.lazy_tag[i])
            self.lazy_tag[i] = 0

    def _make_tag(self, i, val):
        self.lazy_tag[i] = max(self.lazy_tag[i], val)
        return

    def range_ascend(self, left, right, val):
        stack = [(0, self.n - 1, 1)]  # 
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                self._make_tag(i, val)
                continue

            m = s + (t - s) // 2
            self._push_down(i)

            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return

    def point_get(self, ind):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                ans = self.lazy_tag[i]
                break
            m = s + (t - s) // 2
            self._push_down(i)
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return ans

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.lazy_tag[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums


class RangeAddRangeMulSum:

    def __init__(self, n, mod):
        self.n = n
        self.mod = mod
        self.cover = [0] * (4 * n)
        self.cover1 = [0] * (4 * n)
        self.cover2 = [0] * 4 * n
        self.add1 = [0] * (4 * n)  # lazy_tag for mul
        self.add2 = [0] * (4 * n)  # lazy_tag for add
        return

    def _make_tag(self, i, s, t, val, op=1):
        if op == 2:
            self.cover[i] = (self.cover[i] + val * self.cover1[i]) % self.mod
            self.cover2[i] = (self.cover2[i] + (t - s + 1) * val) % self.mod
            self.add2[i] += val
            self.add2[i] %= self.mod
        else:
            self.cover[i] = (self.cover[i] + val * self.cover2[i]) % self.mod
            self.cover1[i] = (self.cover1[i] + (t - s + 1) * val) % self.mod
            self.add1[i] += val
            self.add1[i] %= self.mod
        return

    def _push_up(self, i):
        self.cover[i] = (self.cover[i << 1] + self.cover[(i << 1) | 1]) % self.mod
        self.cover1[i] = (self.cover1[i << 1] + self.cover1[(i << 1) | 1]) % self.mod
        self.cover2[i] = (self.cover2[i << 1] + self.cover2[(i << 1) | 1]) % self.mod
        return

    def _push_down(self, i, s, m, t):
        if self.add1[i]:
            self._make_tag(i << 1, s, m, self.add1[i], op=1)
            self._make_tag((i << 1) | 1, m + 1, t, self.add1[i], op=1)
            self.add1[i] = 0
        if self.add2[i]:
            self._make_tag(i << 1, s, m, self.add2[i], op=2)
            self._make_tag((i << 1) | 1, m + 1, t, self.add2[i], op=2)
            self.add2[i] = 0
        return

    def build(self, nums):

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, s, t, nums[s][0], 1)
                    self._make_tag(i, s, t, nums[s][1], 2)
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

    def range_add_mul(self, left, right, val, op):

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

    def range_sum(self, left, right):
        if left == right:
            s, t, i = 0, self.n - 1, 1
            ans = 0
            while True:
                if left <= s <= t <= right:
                    ans += self.cover[i]
                    ans %= self.mod
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


class RangeAddMulRangeSum:

    def __init__(self, n, mod):
        self.n = n
        self.mod = mod
        self.cover = [0] * (4 * n)
        self.add = [0] * (4 * n)  # lazy_tag for mul
        self.mul = [1] * (4 * n)  # lazy_tag for add
        return

    def _make_tag(self, i, s, t, val, op="add"):
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

    def build(self, nums):

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

    def range_sum(self, left, right):
        # query the range sum
        if left == right:
            s, t, i = 0, self.n - 1, 1
            ans = 0
            while True:
                if left <= s <= t <= right:
                    ans += self.cover[i]
                    ans %= self.mod
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

    def _make_tag(self, i, s, t, val):
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
                nums[s] = self.cover[i] % self.mod
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_affine(self, left, right, val):

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

    def range_sum(self, left, right):
        if left == right:
            s, t, i = 0, self.n - 1, 1
            ans = 0
            while True:
                if left <= s <= t <= right:
                    ans += self.cover[i]
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


class RangeSetRangeSumMinMax:
    def __init__(self, n, initial=math.inf):
        self.n = n
        self.initial = math.inf
        self.cover = [0] * (4 * self.n)  # range sum
        self.lazy_tag = [self.initial] * (4 * self.n)  # because range change can to be 0 the lazy tag must be math.inf
        self.floor = [self.initial] * (
                4 * self.n)  # because range change can to be any integer the floor initial must be math.inf
        self.ceil = [-self.initial] * (
                4 * self.n)  # because range change can to be any integer the ceil initial must be -math.inf
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i] != self.initial:
            self.cover[i << 1] = self.lazy_tag[i] * (m - s + 1)
            self.cover[(i << 1) | 1] = self.lazy_tag[i] * (t - m)

            self.floor[i << 1] = self.lazy_tag[i]
            self.floor[(i << 1) | 1] = self.lazy_tag[i]

            self.ceil[i << 1] = self.lazy_tag[i]
            self.ceil[(i << 1) | 1] = self.lazy_tag[i]

            self.lazy_tag[i << 1] = self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] = self.lazy_tag[i]

            self.lazy_tag[i] = self.initial

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.ceil[i] = max(self.ceil[i << 1], self.ceil[(i << 1) | 1])
        self.floor[i] = min(self.floor[i << 1], self.floor[(i << 1) | 1])
        return

    def _make_tag(self, i, s, t, val):
        self.cover[i] = val * (t - s + 1)
        self.floor[i] = val
        self.ceil[i] = val
        self.lazy_tag[i] = val
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

    def range_set(self, left, right, val):
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

        stack = [(0, self.n - 1, 1)]
        lowest = math.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                lowest = min(lowest, self.floor[i])
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

        stack = [(0, self.n - 1, 1)]
        highest = -math.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = max(highest, self.ceil[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return highest

    def point_get(self, ind):
        s, t, i = 0, self.n - 1, 1  # 
        while True:
            if s == t == ind:
                ans = self.cover[i]
                break
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return ans


class RangeOrRangeOr:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * 4 * n
        self.lazy_tag = [0] * 4 * n
        return

    def _make_tag(self, i, val):
        self.cover[i] |= val
        self.lazy_tag[i] |= val
        return

    def _push_down(self, i):
        if self.lazy_tag[i]:
            self.cover[i << 1] |= self.lazy_tag[i]
            self.cover[(i << 1) | 1] |= self.lazy_tag[i]

            self.lazy_tag[i << 1] |= self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] |= self.lazy_tag[i]

            self.lazy_tag[i] = 0

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] | self.cover[(i << 1) | 1]
        return

    def build(self, nums):

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
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= r:
                    self.cover[i] |= val
                    self.lazy_tag[i] |= val
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

    def range_or_query(self, left, r):
        """query the range or"""
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= r:
                ans |= self.cover[i]
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


class RangeMulRangeMul:
    def __init__(self, n, mod=10 ** 9 + 7):
        self.n = n
        self.cover = [1] * (4 * self.n)  # range sum
        self.lazy_tag = [1] * (4 * self.n)  # lazy tag
        self.mod = mod
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

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i] != 1:
            self.cover[i << 1] = (self.cover[i << 1] * pow(self.lazy_tag[i], m - s + 1, self.mod)) % self.mod
            self.cover[(i << 1) | 1] = (self.cover[(i << 1) | 1] * pow(self.lazy_tag[i], t - m, self.mod)) % self.mod
            self.lazy_tag[i << 1] *= self.lazy_tag[i]
            self.lazy_tag[i << 1] %= self.mod
            self.lazy_tag[(i << 1) | 1] *= self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] %= self.mod
            self.lazy_tag[i] = 1

    def _push_up(self, i):
        self.cover[i] = (self.cover[i << 1] * self.cover[(i << 1) | 1]) % self.mod
        return

    def _make_tag(self, i, s, t, val):
        self.cover[i] = (self.cover[i] * pow(val, (t - s + 1), self.mod))
        self.lazy_tag[i] = (self.lazy_tag[i] * val) % self.mod
        return

    def range_mul_update(self, left, right, val):
        # update the range add

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

    def range_mul_query(self, left, right):
        # query the range sum
        stack = [(0, self.n - 1, 1)]
        ans = 1
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans *= self.cover[i]
                ans %= self.mod
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class RangeSetReverseRangeSumLongestConSub:
    def __init__(self, n):
        self.n = n
        # cover_0, cover_1, sum_1, start_0, start_1, end_0, end_1, s1, t2
        self.cover = [(0,) * 9] * (4 * self.n)  # cover with 0
        self.lazy_tag = [3] * (4 * self.n)  # lazy tag 0-change 1-change 2-reverse
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
        tag = self.lazy_tag[i]
        if val <= 1:
            self.lazy_tag[i] = val
        else:
            if tag <= 1:
                tag = 1 - tag
            elif tag == 2:
                tag = 3
            else:
                tag = 2
            self.lazy_tag[i] = tag
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i] != 3:
            self._make_tag(s, m, i << 1, self.lazy_tag[i])
            self._make_tag(m + 1, t, (i << 1) | 1, self.lazy_tag[i])
            self.lazy_tag[i] = 3

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

    def range_set_reverse(self, left, right, val):
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


class RangeSetRangeSumMinMaxDynamic:
    def __init__(self, n, initial=math.inf):
        # dynamic adding point segment tree in which n can be 1e9
        self.n = n
        self.initial = initial
        self.cover = defaultdict(int)  # range sum must be initial 0
        self.lazy_tag = defaultdict(lambda: self.initial)  # lazy tag must be initial math.inf
        self.floor = defaultdict(int)  # range min can be math.inf
        self.ceil = defaultdict(int)  # range max can be -math.inf
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i] != self.initial:
            self.cover[i << 1] = self.lazy_tag[i] * (m - s + 1)
            self.cover[(i << 1) | 1] = self.lazy_tag[i] * (t - m)

            self.floor[i << 1] = self.lazy_tag[i]
            self.floor[(i << 1) | 1] = self.lazy_tag[i]

            self.ceil[i << 1] = self.lazy_tag[i]
            self.ceil[(i << 1) | 1] = self.lazy_tag[i]

            self.lazy_tag[i << 1] = self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] = self.lazy_tag[i]

            self.lazy_tag[i] = self.initial

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.ceil[i] = max(self.ceil[i << 1], self.ceil[(i << 1) | 1])
        self.floor[i] = min(self.floor[i << 1], self.floor[(i << 1) | 1])
        return

    def _make_tag(self, i, s, t, val):
        self.cover[i] = val * (t - s + 1)
        self.floor[i] = val
        self.ceil[i] = val
        self.lazy_tag[i] = val
        return

    def range_set(self, left, right, val):
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
        stack = [(0, self.n - 1, 1)]
        highest = math.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = min(highest, self.floor[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return highest

    def range_max(self, left, right):

        stack = [(0, self.n - 1, 1)]
        highest = -math.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = max(highest, self.ceil[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return highest

    def range_sum_bisect_left(self, val):
        if val >= self.cover[1]:
            return self.n
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if self.cover[i << 1] > val:
                s, t, i = s, m, i << 1
            else:
                val -= self.cover[i << 1]
                s, t, i = m + 1, t, (i << 1) | 1
        return t


class RangeSetRangeSumMinMaxDynamicDct:
    def __init__(self, n, m=4 * 10 ** 5, initial=math.inf):
        # dynamic adding point segment tree in which n can be 1e9
        self.n = n
        self.initial = initial
        self.cover = [0] * m  # range sum must be initial 0
        self.lazy_tag = [self.initial] * m  # lazy tag must be initial math.inf
        self.floor = [0] * m  # range min can be math.inf
        self.ceil = [0] * m  # range max can be -math.inf
        self.dct = dict()
        self.ind = 1
        return

    def _produce(self, i):
        if i not in self.dct:
            self.dct[i] = self.ind
            self.ind += 1
        while self.ind >= len(self.lazy_tag):
            self.cover.append(0)
            self.lazy_tag.append(self.initial)
            self.floor.append(0)
            self.ceil.append(0)
        return

    def _push_down(self, i, s, m, t):
        self._produce(i)
        self._produce(i << 1)
        self._produce((i << 1) | 1)
        if self.lazy_tag[self.dct[i]] != self.initial:
            self.cover[self.dct[i << 1]] = self.lazy_tag[self.dct[i]] * (m - s + 1)
            self.cover[self.dct[(i << 1) | 1]] = self.lazy_tag[self.dct[i]] * (t - m)

            self.floor[self.dct[i << 1]] = self.lazy_tag[self.dct[i]]
            self.floor[self.dct[(i << 1) | 1]] = self.lazy_tag[self.dct[i]]

            self.ceil[self.dct[i << 1]] = self.lazy_tag[self.dct[i]]
            self.ceil[self.dct[(i << 1) | 1]] = self.lazy_tag[self.dct[i]]

            self.lazy_tag[self.dct[i << 1]] = self.lazy_tag[self.dct[i]]
            self.lazy_tag[self.dct[(i << 1) | 1]] = self.lazy_tag[self.dct[i]]

            self.lazy_tag[self.dct[i]] = self.initial

    def _push_up(self, i):
        self.cover[self.dct[i]] = self.cover[self.dct[i << 1]] + self.cover[self.dct[(i << 1) | 1]]
        self.ceil[self.dct[i]] = max(self.ceil[self.dct[i << 1]], self.ceil[self.dct[(i << 1) | 1]])
        self.floor[self.dct[i]] = min(self.floor[self.dct[i << 1]], self.floor[self.dct[(i << 1) | 1]])
        return

    def _make_tag(self, i, s, t, val):
        self._produce(i)
        self.cover[self.dct[i]] = val * (t - s + 1)
        self.floor[self.dct[i]] = val
        self.ceil[self.dct[i]] = val
        self.lazy_tag[self.dct[i]] = val
        return

    def range_set(self, left, right, val):
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
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            self._produce(i)
            if left <= s and t <= right:
                ans += self.cover[self.dct[i]]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def range_min(self, left, right):
        stack = [(0, self.n - 1, 1)]
        highest = math.inf
        while stack:
            s, t, i = stack.pop()
            self._produce(i)
            if left <= s and t <= right:
                highest = min(highest, self.floor[self.dct[i]])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return highest

    def range_max(self, left, right):
        stack = [(0, self.n - 1, 1)]
        highest = -math.inf
        while stack:
            s, t, i = stack.pop()
            self._produce(i)
            if left <= s and t <= right:
                highest = max(highest, self.ceil[self.dct[i]])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return highest

    def range_sum_bisect_left(self, val):
        if val >= self.cover[1]:
            return self.n
        s, t, i = 0, self.n - 1, 1
        while s < t:
            self._produce(i)
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if self.cover[self.dct[i << 1]] > val:
                s, t, i = s, m, i << 1
            else:
                val -= self.cover[self.dct[i << 1]]
                s, t, i = m + 1, t, (i << 1) | 1
        return t


class RangeSetPreSumMaxDynamic:
    def __init__(self, n, initial=-math.inf):
        # dynamic adding point segment tree in which n can be 1e9
        self.n = n
        self.initial = initial
        self.pre_sum_max = defaultdict(int)
        self.sum = defaultdict(int)
        self.lazy_tag = defaultdict(lambda: self.initial)  # lazy tag must be initial math.inf
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i] != self.initial:
            self._make_tag(i << 1, s, m, self.lazy_tag[i])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy_tag[i])
            self.lazy_tag[i] = self.initial

    def _push_up(self, i):
        self.sum[i] = self.sum[i << 1] + self.sum[(i << 1) | 1]
        self.pre_sum_max[i] = max(self.pre_sum_max[i << 1],
                                  self.sum[i << 1] + max(0, self.pre_sum_max[(i << 1) | 1]))
        return

    def _make_tag(self, i, s, t, val):
        self.pre_sum_max[i] = val * (t - s + 1) if val >= 0 else val
        self.sum[i] = val * (t - s + 1)
        self.lazy_tag[i] = val
        return

    def range_set(self, left, right, val):
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
        if left > right:
            return 0
        stack = [(0, self.n - 1, 1)]
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

    def range_pre_sum_max(self, left):
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        pre_sum = 0
        while stack:
            s, t, i = stack.pop()
            if t <= left:
                ans = max(ans, pre_sum + self.pre_sum_max[i])
                pre_sum += self.sum[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left > m:
                stack.append((m + 1, t, (i << 1) | 1))
            stack.append((s, m, i << 1))
        return ans

    def range_pre_sum_max_bisect_left(self, val):
        s, t, i = 0, self.n - 1, 1
        pre_sum = 0
        while s < t:
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if pre_sum + self.pre_sum_max[i << 1] > val:
                s, t, i = s, m, i << 1
            else:
                pre_sum += self.sum[i << 1]
                s, t, i = m + 1, t, (i << 1) | 1
        if t == self.n - 1 and self.range_pre_sum_max(self.n - 1) <= val:
            return t + 1
        return t


class RangeSetPreSumMaxDynamicDct:
    def __init__(self, n, m=3 * 10 ** 6, initial=-math.inf):
        # dynamic adding point segment tree in which n can be 1e9
        self.n = n
        self.initial = initial
        self.pre_sum_max = [0] * m
        self.sum = [0] * m
        self.lazy_tag = [0] * m  # lazy tag must be initial math.inf
        self.dct = dict()
        self.ind = 1
        return

    def _produce(self, i):
        if i not in self.dct:
            self.dct[i] = self.ind
            self.ind += 1
        while self.ind >= len(self.pre_sum_max):
            self.pre_sum_max.append(0)
            self.sum.append(0)
            self.lazy_tag.append(self.initial)
        return

    def _push_down(self, i, s, m, t):
        self._produce(i)
        if self.lazy_tag[self.dct[i]] != self.initial:
            self._make_tag(i << 1, s, m, self.lazy_tag[self.dct[i]])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy_tag[self.dct[i]])
            self.lazy_tag[self.dct[i]] = self.initial

    def _push_up(self, i):
        self._produce(i)
        self._produce(i << 1)
        self._produce((i << 1) | 1)
        self.sum[self.dct[i]] = self.sum[self.dct[i << 1]] + self.sum[self.dct[(i << 1) | 1]]
        self.pre_sum_max[self.dct[i]] = max(self.pre_sum_max[self.dct[i << 1]],
                                            self.sum[self.dct[i << 1]] + max(0, self.pre_sum_max[
                                                self.dct[(i << 1) | 1]]))
        return

    def _make_tag(self, i, s, t, val):
        self._produce(i)
        self.pre_sum_max[self.dct[i]] = val * (t - s + 1) if val >= 0 else val
        self.sum[self.dct[i]] = val * (t - s + 1)
        self.lazy_tag[self.dct[i]] = val
        return

    def range_set(self, left, right, val):
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
        if left > right:
            return 0
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.sum[self.dct[i]]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def range_pre_sum_max(self, left):
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        pre_sum = 0
        while stack:
            s, t, i = stack.pop()
            self._produce(i)
            if t <= left:
                ans = max(ans, pre_sum + self.pre_sum_max[self.dct[i]])
                pre_sum += self.sum[self.dct[i]]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left > m:
                stack.append((m + 1, t, (i << 1) | 1))
            stack.append((s, m, i << 1))
        return ans

    def range_pre_sum_max_bisect_left(self, val):
        s, t, i = 0, self.n - 1, 1
        pre_sum = 0
        while s < t:
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if pre_sum + self.pre_sum_max[self.dct[i << 1]] > val:
                s, t, i = s, m, i << 1
            else:
                pre_sum += self.sum[self.dct[i << 1]]
                s, t, i = m + 1, t, (i << 1) | 1
        if t == self.n - 1 and self.range_pre_sum_max(self.n - 1) <= val:
            return t + 1
        return t


class RangeKthSmallest:
    def __init__(self, n, k):
        """query the k smallest value of static range which can also change to support dynamic"""
        self.n = n
        self.k = k
        self.cover = [[] for _ in range(4 * self.n)]
        return

    def build(self, nums):

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

    def _push_up(self, i):
        self.cover[i] = self._range_merge_to_disjoint(self.cover[i << 1][:], self.cover[(i << 1) | 1][:])
        return

    def range_kth_smallest(self, left, right):
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
        self.lazy_tag = [0] * 4 * n
        return

    def _make_tag(self, i, val):
        self.cover[i] |= val
        self.lazy_tag[i] |= val
        return

    def _push_down(self, i):
        if self.lazy_tag[i]:
            self.cover[i << 1] |= self.lazy_tag[i]
            self.cover[(i << 1) | 1] |= self.lazy_tag[i]

            self.lazy_tag[i << 1] |= self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] |= self.lazy_tag[i]

            self.lazy_tag[i] = 0

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] & self.cover[(i << 1) | 1]
        return

    def build(self, nums):

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
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= r:
                    self.cover[i] |= val
                    self.lazy_tag[i] |= val
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
        self.lazy_tag = [0] * (4 * self.n)
        return

    def initial(self):
        for i in range(self.n):
            self.cover[i] = 0
            self.lazy_tag[i] = 0
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                    self.lazy_tag[i] = 0
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
                self.lazy_tag[i] = 0
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

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i]:
            self.cover[i << 1] = m - s + 1 - self.cover[i << 1]
            self.cover[(i << 1) | 1] = t - m - self.cover[(i << 1) | 1]

            self.lazy_tag[i << 1] ^= self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] ^= self.lazy_tag[i]

            self.lazy_tag[i] = 0
        return

    def range_reverse(self, left, right):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.cover[i] = t - s + 1 - self.cover[i]
                    self.lazy_tag[i] ^= 1
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

    def point_get(self, ind):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                ans = self.cover[i]
                break
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        return ans

    def range_bit_count(self, left, right):
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

    def range_bit_count_bisect_left(self, val):
        if self.cover[1] < val:
            return -1
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if self.cover[i << 1] >= val:
                s, t, i = s, m, i << 1
            else:
                val -= self.cover[i << 1]
                s, t, i = m + 1, t, (i << 1) | 1
        return t


class RangeRevereRangeAlter:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.lazy_tag = [0] * (4 * self.n)
        self.pre = [0] * (4 * self.n)
        self.suf = [0] * (4 * self.n)
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = 1
                    self.pre[i] = nums[s]
                    self.suf[i] = nums[s]
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
        self.cover[i] = self.cover[i << 1] & self.cover[(i << 1) | 1] & (self.suf[i << 1] ^ self.pre[(i << 1) | 1])
        self.pre[i] = self.pre[i << 1]
        self.suf[i] = self.suf[(i << 1) | 1]
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.suf[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def _push_down(self, i):
        if self.lazy_tag[i]:
            self.pre[i << 1] ^= 1
            self.pre[(i << 1) | 1] ^= 1

            self.suf[i << 1] ^= 1
            self.suf[(i << 1) | 1] ^= 1

            self.lazy_tag[i << 1] ^= 1
            self.lazy_tag[(i << 1) | 1] ^= 1

            self.lazy_tag[i] = 0
        return

    def range_reverse(self, left, right):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self.pre[i] ^= 1
                    self.suf[i] ^= 1
                    self.lazy_tag[i] ^= 1
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

    def range_alter_query(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = -1
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                if not self.cover[i]:
                    return False
                if ans != -1 and ans == self.pre[i]:
                    return False
                ans = self.suf[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return True


class RangeSetRangeOr:
    def __init__(self, n):
        self.n = n
        self.lazy_tag = [math.inf] * (4 * self.n)
        self.cover = [0] * (4 * self.n)
        return

    def _make_tag(self, val, i):
        self.cover[i] = val
        self.lazy_tag[i] = val
        return

    def _push_down(self, i):
        if self.lazy_tag[i] != math.inf:
            self._make_tag(self.lazy_tag[i], i << 1)
            self._make_tag(self.lazy_tag[i], (i << 1) | 1)
            self.lazy_tag[i] = math.inf

    def _push_up(self, i):
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

    def range_set(self, left, right, val):
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

    def range_or(self, left, right):
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


class RangeSetAddRangeSumMinMax:

    def __init__(self, n, initial=1 << 68):
        self.n = n
        self.initial = initial
        self.ceil = [0] * (4 * n)
        self.sum = [0] * (4 * n)
        self.floor = [0] * (4 * n)
        self.set_tag = [-self.initial] * (4 * n)
        self.add_tag = [0] * (4 * n)
        return

    def _make_tag(self, i, s, t, res):
        if res[0] > -self.initial:
            self.ceil[i] = res[0]
            self.floor[i] = res[0]
            self.sum[i] = res[0] * (t - s + 1)
        else:
            self.ceil[i] += res[1]
            self.floor[i] += res[1]
            self.sum[i] += res[1] * (t - s + 1)
        self.set_tag[i], self.add_tag[i] = self._combine_tag(res, (self.set_tag[i], self.add_tag[i]))
        return

    def _combine_tag(self, res1, res2):
        if res1[0] > -self.initial:
            return res1[0], 0
        if res2[0] > -self.initial:
            return res2[0] + res1[1], 0
        return -self.initial, res2[1] + res1[1]

    def _push_up(self, i):
        left, right = self.ceil[i << 1], self.ceil[(i << 1) | 1]
        self.ceil[i] = left if left > right else right

        self.sum[i] = self.sum[i << 1] + self.sum[(i << 1) | 1]

        left, right = self.floor[i << 1], self.floor[(i << 1) | 1]
        self.floor[i] = left if left < right else right
        return

    def _push_down(self, i, s, m, t):
        val = (self.set_tag[i], self.add_tag[i])
        if val[0] > -self.initial or val[1]:
            self._make_tag(i << 1, s, m, val)
            self._make_tag((i << 1) | 1, m + 1, t, val)
            self.set_tag[i], self.add_tag[i] = -self.initial, 0
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, s, t, (-self.initial, nums[s]))
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
                nums[s] = self.sum[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_set_add(self, left, right, val):

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

    def range_max(self, left, right):
        ans = -self.initial
        if left == right:
            s, t, i = 0, self.n - 1, 1
            while True:
                if left <= s <= t <= right:
                    ans = ans if ans > self.ceil[i] else self.ceil[i]
                    break
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                if left <= m:
                    s, t, i = s, m, i << 1
                if right > m:
                    s, t, i = m + 1, t, (i << 1) | 1
            return ans

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = ans if ans > self.ceil[i] else self.ceil[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def range_min(self, left, right):
        ans = self.initial
        if left == right:
            s, t, i = 0, self.n - 1, 1
            while True:
                if left <= s <= t <= right:
                    ans = ans if ans < self.floor[i] else self.floor[i]
                    break
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                if left <= m:
                    s, t, i = s, m, i << 1
                if right > m:
                    s, t, i = m + 1, t, (i << 1) | 1
            return ans

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = ans if ans < self.floor[i] else self.floor[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def range_sum(self, left, right):
        ans = 0
        if left == right:
            s, t, i = 0, self.n - 1, 1
            while True:
                if left <= s <= t <= right:
                    ans += self.sum[i]
                    break
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                if left <= m:
                    s, t, i = s, m, i << 1
                if right > m:
                    s, t, i = m + 1, t, (i << 1) | 1
            return ans

        stack = [(0, self.n - 1, 1)]
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


class PointSetRangeOr:

    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * n)
        return

    def _make_tag(self, i, val):
        self.cover[i] = val
        return

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] | self.cover[(i << 1) | 1]
        return

    def build(self, nums):

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

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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
        return

    def range_or(self, left, right):
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

    def range_or_bisect_right(self, left, right, k):
        stack = [(0, self.n - 1, 1)]
        ans = val = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and (self.cover[i] | val).bit_count() <= k:
                val |= self.cover[i]
                ans = t
                continue
            if s == t:
                break
            m = s + (t - s) // 2
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans


class RangeSetPreSumMax:
    def __init__(self, n, initial=-math.inf):
        # dynamic adding point segment tree in which n can be 1e9
        self.n = n
        self.initial = initial
        self.pre_sum_max = [0] * 4 * n
        self.sum = [0] * 4 * n
        self.lazy_tag = [initial] * 4 * n  # lazy tag must be initial math.inf
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.sum[i] = nums[s]
                    self.pre_sum_max[i] = nums[s]
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

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i] != self.initial:
            self._make_tag(i << 1, s, m, self.lazy_tag[i])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy_tag[i])
            self.lazy_tag[i] = self.initial

    def _push_up(self, i):
        self.sum[i] = self.sum[i << 1] + self.sum[(i << 1) | 1]
        self.pre_sum_max[i] = max(self.pre_sum_max[i << 1],
                                  self.sum[i << 1] + max(0, self.pre_sum_max[(i << 1) | 1]))
        return

    def _make_tag(self, i, s, t, val):
        self.pre_sum_max[i] = val * (t - s + 1) if val >= 0 else val
        self.sum[i] = val * (t - s + 1)
        self.lazy_tag[i] = val
        return

    def range_set(self, left, right, val):
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
        if left > right:
            return 0
        stack = [(0, self.n - 1, 1)]
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

    def range_pre_sum_max(self, left):
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        pre_sum = 0
        while stack:
            s, t, i = stack.pop()
            if t <= left:
                ans = max(ans, pre_sum + self.pre_sum_max[i])
                pre_sum += self.sum[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left > m:
                stack.append((m + 1, t, (i << 1) | 1))
            stack.append((s, m, i << 1))
        return ans

    def range_pre_sum_max_range(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        pre_sum = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = max(ans, pre_sum + self.pre_sum_max[i])
                pre_sum += self.sum[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans

    def range_pre_sum_max_bisect_left(self, val):
        s, t, i = 0, self.n - 1, 1
        pre_sum = 0
        while s < t:
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if pre_sum + self.pre_sum_max[i << 1] > val:
                s, t, i = s, m, i << 1
            else:
                pre_sum += self.sum[i << 1]
                s, t, i = m + 1, t, (i << 1) | 1
        if t == self.n - 1 and self.range_pre_sum_max(self.n - 1) <= val:
            return t + 1
        return t


class PointSetRangeXor:

    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * n)
        return

    def _make_tag(self, i, val):
        self.cover[i] = val
        return

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] ^ self.cover[(i << 1) | 1]
        return

    def build(self, nums):

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

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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
        return

    def range_xor(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans ^= self.cover[i]
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class RangeModPointSetRangeSum:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.ceil = [0] * (4 * self.n)
        return

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.ceil[i] = max(self.ceil[i << 1], self.ceil[(i << 1) | 1])
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                    self.ceil[i] = nums[s]
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_mod(self, left, right, mod):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if self.cover[i] < mod:
                    continue
                if s == t:
                    self.cover[i] %= mod
                    self.ceil[i] = self.cover[i]
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                if left <= m and self.cover[i << 1] >= mod:
                    stack.append((s, m, i << 1))
                if right > m and self.cover[(i << 1) | 1] >= mod:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                self.cover[i] = val
                self.ceil[i] = val
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


class PointSetRangeComposite:

    def __init__(self, n, mod, m=32):
        self.n = n
        self.mod = mod
        self.m = m
        self.mask = (1 << m) - 1
        self.cover = [0] * (4 * n)
        return

    def _make_tag(self, i, val):
        self.cover[i] = val
        return

    def _push_up(self, i):
        self.cover[i] = self._merge_cover(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def _merge_cover(self, val1, val2):
        mul1, add1 = val1 >> self.m, val1 & self.mask
        mul2, add2 = val2 >> self.m, val2 & self.mask
        return ((mul2 * mul1 % self.mod) << self.m) | ((mul2 * add1 + add2) % self.mod)

    def build(self, nums):

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

    def range_composite(self, left, right):
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


class RangeSetRangeMaxNonEmpConSubSum:
    def __init__(self, n, initial=math.inf):
        self.n = n
        self.initial = initial
        self.cover = [-initial] * (4 * self.n)
        self.left = [-initial] * (4 * self.n)
        self.right = [-initial] * (4 * self.n)
        self.lazy_tag = [initial] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        return

    def _make_tag(self, i, s, t, val):
        self.cover[i] = val * (t - s + 1) if val > 0 else val
        self.left[i] = val * (t - s + 1) if val > 0 else val
        self.right[i] = val * (t - s + 1) if val > 0 else val
        self.sum[i] = val * (t - s + 1)
        self.lazy_tag[i] = val
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i] != self.initial:
            self._make_tag(i << 1, s, m, self.lazy_tag[i])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy_tag[i])
            self.lazy_tag[i] = self.initial
        return

    def _range_merge_to_disjoint(self, res1, res2):
        res = [0] * 4
        res[0] = max(res1[0], res2[0])
        res[0] = max(res[0], res1[2] + res2[1])
        res[1] = max(res1[1], res1[3] + res2[1])
        res[2] = max(res2[2], res2[3] + res1[2])
        res[3] = res1[3] + res2[3]
        return res

    def _push_up(self, i):
        res1 = self.cover[i << 1], self.left[i << 1], self.right[i << 1], self.sum[i << 1]
        res2 = self.cover[(i << 1) | 1], self.left[(i << 1) | 1], self.right[(i << 1) | 1], self.sum[(i << 1) | 1]
        self.cover[i], self.left[i], self.right[i], self.sum[i] = self._range_merge_to_disjoint(res1, res2)
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, s, t, nums[s])
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
                    nums[s] = self.sum[i]
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return nums

    def range_set(self, left, right, val):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, s, t, val)
                    continue
                m = s + (t - s) // 2
                stack.append((s, t, ~i))
                self._push_down(i, s, m, t)
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                self._make_tag(i, s, t, val)
                break
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        while i > 1:
            i //= 2
            self._push_up(i)
        return

    def range_max_non_emp_con_sub_sum(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = [-self.initial] * 4
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                cur = [self.cover[i], self.left[i], self.right[i], self.sum[i]]
                ans = self._range_merge_to_disjoint(cur, ans)
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans[0]


class RangeSetRangeSegCountLength:
    def __init__(self, n, initial=-1):
        self.n = n
        self.initial = initial
        self.cover = [0] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        self.left = [0] * (4 * self.n)
        self.right = [0] * (4 * self.n)
        self.lazy_tag = [self.initial] * (4 * self.n)
        return

    def _make_tag(self, i, s, t, val):
        self.cover[i] = val
        self.left[i] = val
        self.right[i] = val
        self.sum[i] = val * (t - s + 1)
        self.lazy_tag[i] = val
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i] != self.initial:
            self._make_tag(i << 1, s, m, self.lazy_tag[i])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy_tag[i])
            self.lazy_tag[i] = self.initial
        return

    @classmethod
    def _range_merge_to_disjoint(cls, res1, res2):
        res = [0] * 4
        res[0] = res1[0] + res2[0]
        res[1] = res1[1]
        res[2] = res2[2]
        res[3] = res1[3] + res2[3]
        if res1[2] and res2[1]:
            res[0] -= 1
        return res

    def _push_up(self, i):
        res1 = self.cover[i << 1], self.left[i << 1], self.right[i << 1], self.sum[i << 1]
        res2 = self.cover[(i << 1) | 1], self.left[(i << 1) | 1], self.right[(i << 1) | 1], self.sum[(i << 1) | 1]
        self.cover[i], self.left[i], self.right[i], self.sum[i] = self._range_merge_to_disjoint(res1, res2)
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, s, t, nums[s])
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
                    nums[s] = self.sum[i]
                    continue
                stack.append((s, t, ~i))
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                stack.append((s, m, i << 1))
                stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return nums

    def range_set(self, left, right, val):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, s, t, val)
                    continue
                m = s + (t - s) // 2
                stack.append((s, t, ~i))
                self._push_down(i, s, m, t)
                if left <= m:
                    stack.append((s, m, i << 1))
                if right > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        assert i == 1
        return

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                self._make_tag(i, s, t, val)
                break
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        while i > 1:
            i //= 2
            self._push_up(i)
        return

    def range_seg_count_length(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = [0] * 4
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                cur = [self.cover[i], self.left[i], self.right[i], self.sum[i]]
                ans = self._range_merge_to_disjoint(cur, ans)
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans[0], ans[-1]


class PointSetRangeLongestAlter:
    def __init__(self, n):
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
    def _min(a, b):
        return a if a < b else b

    def _push_up(self, i, s, m, t):
        self.cover[i] = max(self.cover[i << 1], self.cover[(i << 1) | 1])
        self.cover[i] = max(self.cover[i], self.right_0[i << 1] + self.left_1[(i << 1) | 1])
        self.cover[i] = max(self.cover[i], self.right_1[i << 1] + self.left_0[(i << 1) | 1])

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

    def build(self):
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

    def point_set_range_longest_alter(self, left, right):
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
                val = self.cover[i]
                nums[s] = val
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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

    def range_max(self, left, right):
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

    def range_max_bisect_left(self, left, right, val):
        stack = [(0, self.n - 1, 1)]
        res = -1
        while stack and res == -1:
            s, t, i = stack.pop()
            if s == t:
                if left <= s <= right and self.cover[i] >= val:
                    res = s
                continue
            m = s + (t - s) // 2
            if right > m and self.cover[(i << 1) | 1] >= val:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m and self.cover[i << 1] >= val:
                stack.append((s, m, i << 1))
        return res


class PointSetRangeMaxSecondCnt:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.first = [(initial, 0)] * (4 * n)
        self.second = [(initial, 0)] * (4 * n)
        return

    def merge(self, first1, second1, first2, second2):
        tmp = [first1, second1, first2, second2]
        max1 = max2 = self.initial
        cnt1 = cnt2 = 0
        for a, b in tmp:
            if a > max1:
                max2, cnt2 = max1, cnt1
                max1, cnt1 = a, b
            elif a == max1:
                cnt1 += b
            elif a > max2:
                max2, cnt2 = a, b
            elif a == max2:
                cnt2 += b
        return (max1, cnt1), (max2, cnt2)

    def push_up(self, i):
        self.first[i], self.second[i] = self.merge(self.first[i << 1], self.second[i << 1], self.first[(i << 1) | 1],
                                                   self.second[(i << 1) | 1])
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.first[i] = (nums[s], 1)
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self.push_up(i)
        return

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                val = self.first[i][0]
                nums[s] = val
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                self.first[i] = (val, 1)
                break
            m = s + (t - s) // 2
            if ind <= m:
                s, t, i = s, m, i << 1
            else:
                s, t, i = m + 1, t, (i << 1) | 1
        while i > 1:
            i //= 2
            self.push_up(i)
        return

    def range_max_second_cnt(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans1 = (self.initial, 0)
        ans2 = (self.initial, 0)
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans1, ans2 = self.merge(ans1, ans2, self.first[i], self.second[i])
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans1 + ans2


class RangeAddRangeMaxIndex:
    def __init__(self, n):
        self.n = n
        self.lazy_tag = [0] * (4 * self.n)  # lazy tag
        self.ceil = [0] * (4 * self.n)  # range max
        self.index = [0] * (4 * self.n)  # minimum index of range max
        return

    def build(self, nums):

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.ceil[i] = nums[s]
                    self.index[i] = s
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def _push_down(self, i):
        if self.lazy_tag[i]:
            self.ceil[i << 1] += self.lazy_tag[i]
            self.ceil[(i << 1) | 1] += self.lazy_tag[i]

            self.lazy_tag[i << 1] += self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] += self.lazy_tag[i]

            self.lazy_tag[i] = 0

    def _push_up(self, i):
        if self.ceil[i << 1] >= self.ceil[(i << 1) | 1]:
            self.ceil[i] = self.ceil[i << 1]
            self.index[i] = self.index[i << 1]
        else:
            self.ceil[i] = self.ceil[(i << 1) | 1]
            self.index[i] = self.index[(i << 1) | 1]
        return

    def _make_tag(self, i, val):
        self.ceil[i] += val
        self.lazy_tag[i] += val
        return

    def range_add(self, left, right, val):
        # update the range add
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, val)
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

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                val = self.ceil[i]
                nums[s] = val
                continue
            self._push_down(i)
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_max_bisect_left(self, left, right, val):
        stack = [(0, self.n - 1, 1)]
        res = -1
        while stack and res == -1:
            s, t, i = stack.pop()
            if s == t:
                if left <= s <= right and self.ceil[i] >= val:
                    res = s
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if right > m and self.ceil[(i << 1) | 1] >= val:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m and self.ceil[i << 1] >= val:
                stack.append((s, m, i << 1))
        return res

    def range_max(self, left, right):
        # query the rang max
        stack = [(0, self.n - 1, 1)]
        highest = -math.inf
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                highest = max(highest, self.ceil[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return highest

    def range_max_index(self, left, right):
        stack = [(0, self.n - 1, 1)]
        highest = -math.inf
        ind = -1
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                if self.ceil[i] > highest:
                    highest = self.ceil[i]
                    ind = self.index[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i)
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return highest, ind


class PointSetRangeMaxIndex:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover = [initial] * (4 * n)
        self.index = [initial] * (4 * n)
        return

    @classmethod
    def merge(cls, x, y):
        return x if x > y else y

    def _push_up(self, i):
        a, b = self.cover[i << 1], self.cover[(i << 1) | 1]
        if a > b:
            self.cover[i], self.index[i] = a, self.index[i << 1]
        else:
            self.cover[i], self.index[i] = b, self.index[(i << 1) | 1]
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s]
                    self.index[i] = s
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

    def point_set_index(self, ind, x, val):
        s, t, i = 0, self.n - 1, 1
        while True:
            if s == t == ind:
                self.cover[i] = val
                self.index[i] = x
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

    def range_max_index(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        ind = -1
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                if ans < self.cover[i]:
                    ans, ind = self.cover[i], self.index[i]
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans, ind


class PointSetRangeSum:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover = [initial] * (4 * n)
        return

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        return

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
        s, t, i = 0, self.n - 1, 1
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

    def range_sum_bisect_left(self, val):
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            if self.cover[i << 1] >= val:
                s, t, i = s, m, i << 1
            else:
                val -= self.cover[i << 1]
                s, t, i = m + 1, t, (i << 1) | 1
        return t

    def range_sum_bisect_right(self, val):
        s, t, i = 0, self.n - 1, 1
        while s < t:
            m = s + (t - s) // 2
            if self.cover[i << 1] > val:
                s, t, i = s, m, i << 1
            else:
                val -= self.cover[i << 1]
                s, t, i = m + 1, t, (i << 1) | 1
        return t


class PointSetRangeInversion:

    def __init__(self, n, m=40, initial=0):
        self.n = n
        self.m = m
        self.initial = initial
        self.cover = [initial] * 4 * n
        self.cnt = [initial] * (4 * n * m)
        return

    def _merge(self, lst1, lst2):
        ans = pre = 0
        lst = [0] * self.m
        for i in range(self.m):
            ans += pre * lst1[i]
            lst[i] = lst1[i] + lst2[i]
            pre += lst2[i]
        return ans, lst

    def _push_up(self, i):
        lst1 = self.cnt[(i << 1) * self.m: (i << 1) * self.m + self.m]
        lst2 = self.cnt[((i << 1) | 1) * self.m: ((i << 1) | 1) * self.m + self.m]
        ans, lst = self._merge(lst1, lst2)
        self.cover[i] = ans + self.cover[i << 1] + self.cover[(i << 1) | 1]
        self.cnt[i * self.m:i * self.m + self.m] = lst
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cnt[i * self.m + nums[s]] = 1
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
                for j in range(self.m):
                    if self.cnt[i * self.m + j] == 1:
                        nums[s] = j
                        break
                continue
            m = s + (t - s) // 2
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
        lst = [0] * self.m
        lst[val] = 1
        while True:
            if s == t == ind:
                self.cnt[i * self.m:i * self.m + self.m] = lst[:]
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

    def range_inverse(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        lst = [0] * self.m
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                res, lst = self._merge(lst, self.cnt[i * self.m:(i + 1) * self.m])
                ans += res + self.cover[i]
                continue
            m = s + (t - s) // 2
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans

    def range_inverse_count(self, left, right):
        stack = [(0, self.n - 1, 1)]
        lst = [0] * self.m
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                _, lst = self._merge(lst, self.cnt[i * self.m:(i + 1) * self.m])
                continue
            m = s + (t - s) // 2
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return lst


class MatrixBuildRangeMul:

    def __init__(self, n, initial=0, mod=10 ** 9 + 7):
        self.n = n
        self.initial = initial
        self.mod = mod
        self.cover1 = [initial] * (4 * n)
        self.cover2 = [initial] * (4 * n)
        self.cover3 = [initial] * (4 * n)
        self.cover4 = [initial] * (4 * n)
        return

    def _push_up(self, i):
        lst1 = (self.cover1[i << 1], self.cover2[i << 1],
                self.cover3[i << 1], self.cover4[i << 1])
        lst2 = (self.cover1[(i << 1) | 1], self.cover2[(i << 1) | 1],
                self.cover3[(i << 1) | 1], self.cover4[(i << 1) | 1])
        self.cover1[i], self.cover2[i], self.cover3[i], self.cover4[i] = self._merge(lst1, lst2)
        return

    def _merge(self, lst1, lst2):
        a1, a2, a3, a4 = lst1
        b1, b2, b3, b4 = lst2
        ab1 = a1 * b1 + a2 * b3
        ab2 = a1 * b2 + a2 * b4
        ab3 = a3 * b1 + a4 * b3
        ab4 = a3 * b2 + a4 * b4
        return ab1 % self.mod, ab2 % self.mod, ab3 % self.mod, ab4 % self.mod

    def matrix_build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    a, b, c, d = nums[s * 4], nums[s * 4 + 1], nums[s * 4 + 2], nums[s * 4 + 3]
                    self.cover1[i], self.cover2[i], self.cover3[i], self.cover4[i] = a, b, c, d
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def range_mul(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = (1, 0, 0, 1)
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = self._merge(ans, (self.cover1[i], self.cover2[i], self.cover3[i], self.cover4[i]))
                continue
            m = s + (t - s) // 2
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans


class RangeXorUpdateRangeXorQuery:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (4 * self.n)
        self.lazy_tag = [0] * (4 * self.n)
        return

    def _push_up(self, i):
        self.cover[i] = self.cover[i << 1] ^ self.cover[(i << 1) | 1]
        return

    def _make_tag(self, i, s, t, val):
        if (t - s + 1) % 2:
            self.cover[i] ^= val
        self.lazy_tag[i] ^= val
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i]:
            self._make_tag(i << 1, s, m, self.lazy_tag[i])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy_tag[i])
            self.lazy_tag[i] = 0

    def build(self, nums):
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

    def range_xor_update(self, left, right, val):
        if left == right:

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

    def range_xor_query(self, left, right):
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


class PointSetRangeMin:

    def __init__(self, n, initial=math.inf):
        self.n = n
        self.initial = initial
        self.cover = [initial] * (4 * n)
        return

    def _make_tag(self, i, val):
        self.cover[i] = val
        return

    def _push_up(self, i):
        self.cover[i] = min(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def build(self, nums):

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

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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
        return

    def range_min(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = self.initial
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = min(ans, self.cover[i])
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans

    def range_min_bisect_left(self, left, right, val):
        stack = [(0, self.n - 1, 1)]
        res = -1
        while stack and res == -1:
            a, b, i = stack.pop()
            if a == b:
                if left <= a <= right and self.cover[i] <= val:
                    res = a
                continue
            m = a + (b - a) // 2
            if m + 1 <= right and self.cover[(i << 1) | 1] <= val:
                stack.append((m + 1, b, (i << 1) | 1))
            if left <= m and self.cover[i << 1] <= val:
                stack.append((a, m, i << 1))
        return res


class RangeAddRangeMinCount:
    def __init__(self, n):
        self.n = n
        self.lazy_tag = [0] * (4 * self.n)  # lazy tag
        self.floor = [0] * (4 * self.n)  # range min
        self.cnt = [0] * (4 * self.n)  # range max
        return

    def build(self, nums):

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self._make_tag(i, s, t, nums[s])
                    self.cnt[i] = 1
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i)
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i]:
            self.floor[i << 1] += self.lazy_tag[i]
            self.floor[(i << 1) | 1] += self.lazy_tag[i]

            self.lazy_tag[i << 1] += self.lazy_tag[i]
            self.lazy_tag[(i << 1) | 1] += self.lazy_tag[i]

            self.lazy_tag[i] = 0

    def _push_up(self, i):
        self.floor[i] = min(self.floor[i << 1], self.floor[(i << 1) | 1])
        cur = 0
        if self.floor[i] == self.floor[i << 1]:
            cur += self.cnt[i << 1]
        if self.floor[i] == self.floor[(i << 1) | 1]:
            cur += self.cnt[(i << 1) | 1]
        self.cnt[i] = cur
        return

    def _make_tag(self, i, s, t, val):
        self.floor[i] += val
        self.lazy_tag[i] += val
        return

    def range_add(self, left, right, val):
        # update the range add

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

    def get(self):
        stack = [(0, self.n - 1, 1)]
        nums = [0] * self.n
        while stack:
            s, t, i = stack.pop()
            if s == t:
                nums[s] = self.floor[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            stack.append((s, m, i << 1))
            stack.append((m + 1, t, (i << 1) | 1))
        return nums

    def range_min_count(self, left, right):
        stack = [(0, self.n - 1, 1)]
        res = math.inf
        cnt = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                if self.floor[i] < res:
                    res = self.floor[i]
                    cnt = self.cnt[i]
                elif self.floor[i] == res:
                    cnt += self.cnt[i]
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return res, cnt


class PointSetRangeMaxMinGap:

    def __init__(self, n, initial=math.inf):
        self.n = n
        self.initial = initial
        self.ceil = [-initial] * (4 * n)
        self.floor = [initial] * (4 * n)
        return

    def push_up(self, i):
        a, b = self.ceil[i << 1], self.ceil[(i << 1) | 1]
        self.ceil[i] = a if a > b else b
        a, b = self.floor[i << 1], self.floor[(i << 1) | 1]
        self.floor[i] = a if a < b else b
        return

    def build(self, nums):
        for i in range(self.n):
            self.ceil[i + self.n] = nums[i]
            self.floor[i + self.n] = nums[i]
        for i in range(self.n - 1, 0, -1):
            self.push_up(i)
        return

    def get(self):
        return self.ceil[self.n:]

    def point_set(self, ind, val):
        ind += self.n
        self.ceil[ind] = self.floor[ind] = val
        while ind > 1:
            ind //= 2
            self.push_up(ind)
        return

    def range_max_min_gap(self, left, right):
        ceil_left = ceil_right = -self.initial
        left += self.n
        right += self.n + 1
        floor_left = floor_right = self.initial
        while left < right:
            if left & 1:
                ceil_left = ceil_left if ceil_left > self.ceil[left] else self.ceil[left]
                floor_left = floor_left if floor_left < self.floor[left] else self.floor[left]
                left += 1
            if right & 1:
                right -= 1
                ceil_right = ceil_right if ceil_right > self.ceil[right] else self.ceil[right]
                floor_right = floor_right if floor_right < self.floor[right] else self.floor[right]
            left >>= 1
            right >>= 1
        ceil = ceil_right if ceil_right > ceil_left else ceil_left
        floor = floor_right if floor_right < floor_left else floor_left
        return ceil - floor


class PointSetRangeMinCount:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.m = n.bit_length()
        self.mask = (1 << self.m) - 1
        self.cover = [initial] * (4 * n)
        return

    def _make_tag(self, i, val):
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

    def build(self, nums):

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

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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
        return

    def range_min_count(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = (math.inf << self.m) | 1
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

        a = -math.inf
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
        return self.cover[1]


class PointSetRangeAscendSubCnt:
    def __init__(self, lst):
        self.n = len(lst)
        self.lst = lst
        self.pref = [0] * 4 * self.n
        self.suf = [0] * 4 * self.n
        self.cover = [0] * 4 * self.n
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

    def _merge(self, lst1, lst2, m):
        pref1, suf1, cover1, length1 = lst1
        pref2, suf2, cover2, length2 = lst2
        pref = pref1
        if pref1 == length1 and self.lst[m] <= self.lst[m + 1]:
            pref += pref2

        suf = suf2
        if suf2 == length2 and self.lst[m] <= self.lst[m + 1]:
            suf += suf1

        cover = cover1 + cover2
        if self.lst[m] <= self.lst[m + 1]:
            x, y = suf1, pref2
            cover -= (x + 1) * x // 2 + (y + 1) * y // 2
            cover += (x + y + 1) * (x + y) // 2

        return pref, suf, cover, length1 + length2, m + length2

    def _push_up(self, i, s, t):
        m = s + (t - s) // 2
        lst1 = (self.pref[i << 1], self.suf[i << 1], self.cover[i << 1], m - s + 1)
        lst2 = (self.pref[(i << 1) | 1], self.suf[(i << 1) | 1], self.cover[(i << 1) | 1], t - m)
        self.pref[i], self.suf[i], self.cover[i], _, _ = self._merge(lst1, lst2, m)
        return

    def point_set(self, x, val):
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
        return

    def range_ascend_sub_cnt(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = (0, 0, 0, 0)
        mid = -1
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                if mid == -1:
                    ans = (self.pref[i], self.suf[i], self.cover[i], t - s + 1)
                else:
                    ans = self._merge(ans, [self.pref[i], self.suf[i], self.cover[i], t - s + 1], mid)[:-1]
                mid = t
                continue
            m = s + (t - s) // 2
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans[2]


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


class RangeDivideRangeSum:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.all_factor_cnt = [0, 1] + [2 for _ in range(2, m + 1)]
        for i in range(2, self.m + 1):
            x = i
            while x * i <= self.m:
                self.all_factor_cnt[x * i] += 1
                if i != x:
                    self.all_factor_cnt[x * i] += 1
                x += 1
        self.cover = [0] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        return

    def build(self, nums):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = nums[s] if nums[s] > 2 else 1
                    self.sum[i] = nums[s]
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
                self.sum[i] = self.sum[i << 1] + self.sum[(i << 1) | 1]
        return

    def range_divide(self, left, right):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if self.cover[i] == t - s + 1:
                    continue
                if s == t:
                    nex = self.all_factor_cnt[self.cover[i]]
                    self.sum[i] = nex
                    self.cover[i] = nex if nex > 2 else 1
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
                self.sum[i] = self.sum[i << 1] + self.sum[(i << 1) | 1]
        return

    def range_sum(self, left, right):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans += self.sum[i]
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans


class PointSetRangeMaxSubSum:
    def __init__(self, n, initial=math.inf):
        self.n = n
        self.initial = initial
        self.cover = [-initial] * (4 * self.n)
        self.left = [-initial] * (4 * self.n)
        self.right = [-initial] * (4 * self.n)
        self.sum = [0] * (4 * self.n)
        return

    def build(self, nums):

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

    def _make_tag(self, i, val):
        self.cover[i] = val
        self.left[i] = val
        self.right[i] = val
        self.sum[i] = val
        return

    def _range_merge_to_disjoint(self, res1, res2):
        res = [0] * 4
        res[0] = max(res1[0], res2[0])
        res[0] = max(res[0], res1[2] + res2[1])
        res[1] = max(res1[1], res1[3] + res2[1])
        res[2] = max(res2[2], res2[3] + res1[2])
        res[3] = res1[3] + res2[3]
        return res

    def _push_up(self, i):
        res1 = self.cover[i << 1], self.left[i << 1], self.right[i << 1], self.sum[i << 1]
        res2 = self.cover[(i << 1) | 1], self.left[(i << 1) | 1], self.right[(i << 1) | 1], self.sum[(i << 1) | 1]
        self.cover[i], self.left[i], self.right[i], self.sum[i] = self._range_merge_to_disjoint(res1, res2)
        return

    def point_set_range_max_sub_sum(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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


class PointSetRangeNotExistABC:
    def __init__(self, n):
        self.n = n
        self.a = [0] * 4 * self.n
        self.b = [0] * 4 * self.n
        self.c = [0] * 4 * self.n
        self.ab = [0] * 4 * self.n
        self.bc = [0] * 4 * self.n
        self.abc = [0] * 4 * self.n
        return

    def build(self, nums):
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

    def _make_tag(self, i, val):
        if val == "a":
            self.a[i] = 1
            self.b[i] = 0
            self.c[i] = 0
        elif val == "b":
            self.a[i] = 0
            self.b[i] = 1
            self.c[i] = 0
        else:
            self.a[i] = 0
            self.b[i] = 0
            self.c[i] = 1
        return

    def _push_up(self, i):
        i1 = i << 1
        i2 = (i << 1) | 1
        a1, b1, c1, ab1, bc1, abc1 = self.a[i1], self.b[i1], self.c[i1], self.ab[i1], self.bc[i1], self.abc[i1]
        a2, b2, c2, ab2, bc2, abc2 = self.a[i2], self.b[i2], self.c[i2], self.ab[i2], self.bc[i2], self.abc[i2]
        a = a1 + a2
        b = b1 + b2
        c = c1 + c2
        ab = min(ab1 + b2, a1 + ab2)
        bc = min(bc1 + c2, b1 + bc2)
        abc = min(abc1 + c2, ab1 + bc2, a1 + abc2)
        self.a[i], self.b[i], self.c[i], self.ab[i], self.bc[i], self.abc[i] = a, b, c, ab, bc, abc
        return

    def point_set(self, ind, val):
        s, t, i = 0, self.n - 1, 1
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
        return self.abc[1]


class PointSetRandomRangeMode:

    def __init__(self, nums: List[int], guess_round=20):
        self.n = len(nums)
        self.dct = [SortedList() for _ in range(self.n + 1)]
        for i, num in enumerate(nums):
            self.dct[num].add(i)
        self.nums = nums[:]
        self.guess_round = guess_round

    def point_set(self, i, val):
        self.dct[self.nums[i]].discard(i)
        self.nums[i] = val
        self.dct[self.nums[i]].add(i)
        return

    def range_mode(self, left: int, right: int, threshold=-1) -> int:
        if threshold == -1:
            threshold = (right - left + 1) / 2
        for _ in range(self.guess_round):
            num = self.nums[random.randint(left, right)]
            cur = self.dct[num].bisect_right(right) - self.dct[num].bisect_left(left)
            if cur > threshold:
                return num
        return -1


class PointSetBitRangeMode:
    def __init__(self, nums: List[int], m=20):
        self.n = len(nums)
        self.m = m
        self.pre = [PointAddRangeSum(self.n) for _ in range(self.m)]
        self.dct = [SortedList() for _ in range(self.n + 1)]
        for i, num in enumerate(nums):
            self.dct[num].add(i)
        for i in range(self.m):
            self.pre[i].build([int(num & (1 << i) > 0) for num in nums])
        self.nums = nums[:]
        return

    def point_set(self, i, val):
        num = self.nums[i]
        self.dct[num].discard(i)

        for j in range(self.m):
            if num & (1 << j):
                self.pre[j].point_add(i + 1, -1)
        self.nums[i] = val
        self.dct[val].add(i)
        for j in range(self.m):
            if val & (1 << j):
                self.pre[j].point_add(i + 1, 1)
        return

    def range_mode(self, left: int, right: int, threshold=-1) -> int:
        if threshold == -1:
            threshold = (right - left + 1) / 2
        val = 0
        for j in range(self.m):
            if self.pre[j].range_sum(left + 1, right + 1) > threshold:
                val |= (1 << j)
        ans = self.dct[val].bisect_right(right) - self.dct[val].bisect_left(left)
        return val if ans > threshold else -1


class PointSetMergeRangeMode:

    def __init__(self, nums):
        self.n = len(nums)
        self.dct = [SortedList() for _ in range(self.n + 1)]
        for i, num in enumerate(nums):
            self.dct[num].add(i)
        self.cover = [0] * 4 * self.n
        self.nums = nums[:]
        self.build()
        return

    def _push_up(self, i, s, t):
        a = self.cover[i << 1]
        b = self.cover[(i << 1) | 1]
        cnt_a = self.dct[a].bisect_right(t) - self.dct[a].bisect_left(s)
        cnt_b = self.dct[b].bisect_right(t) - self.dct[b].bisect_left(s)
        self.cover[i] = a if cnt_a > cnt_b else b
        return

    def build(self):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t:
                    self.cover[i] = self.nums[s]
                else:
                    stack.append((s, t, ~i))
                    m = s + (t - s) // 2
                    stack.append((s, m, i << 1))
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i, s, t)
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
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if s == t == ind:
                    self.dct[self.nums[ind]].discard(ind)
                    self.nums[ind] = val
                    self.dct[self.nums[ind]].add(ind)
                    self.cover[i] = val
                    continue

                m = s + (t - s) // 2
                stack.append((s, t, ~i))

                if ind <= m:
                    stack.append((s, m, i << 1))
                if ind > m:
                    stack.append((m + 1, t, (i << 1) | 1))
            else:
                i = ~i
                self._push_up(i, s, t)

    def range_mode(self, left, right, threshold=-1):
        if threshold == -1:
            threshold = (right - left + 1) / 2
        stack = [(0, self.n - 1, 1)]
        ans = -1
        cnt = 0
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                if ans != -1:
                    ans = self.cover[i]
                    cnt = self.dct[ans].bisect_right(right) - self.dct[ans].bisect_left(left)
                else:
                    num = self.cover[i]
                    cur_cnt = self.dct[num].bisect_right(right) - self.dct[num].bisect_left(left)
                    if cur_cnt > cnt:
                        ans = num
                        cnt = cur_cnt
                if cnt > threshold:
                    return ans
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return -1


class SegmentTreeOptBuildGraph:

    def __init__(self, n):
        self.n = n
        self.edges = []
        self.leaves = []
        self.build()
        return

    def build(self):
        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if s == t:
                self.leaves.append(i)
                continue
            m = s + (t - s) // 2
            stack.append((m + 1, t, (i << 1) | 1))
            stack.append((s, m, i << 1))
            self.edges.append((i, i << 1))
            self.edges.append((i, (i << 1) | 1))
        return

    def range_opt(self, left, right):
        assert 0 <= left <= right < self.n
        stack = [(0, self.n - 1, 1)]
        ans = []
        while stack:
            s, t, i = stack.pop()
            if left <= s <= t <= right:
                ans.append(i)
                continue
            if s == t:
                continue
            m = s + (t - s) // 2
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
            if left <= m:
                stack.append((s, m, i << 1))
        return ans


class SegmentTreeOptBuildGraphZKW:

    def __init__(self, n, build=False):
        self.n = n
        self.edges = []
        if build:
            self.build()
        return

    def build(self):
        for i in range(self.n - 1, 0, -1):
            self.edges.append((i, i << 1))
            self.edges.append((i, (i << 1) | 1))
        return

    def range_opt(self, left, right):
        assert 0 <= left <= right < self.n
        left += self.n
        ans = []
        right += self.n + 1
        while left < right:
            if left & 1:
                ans.append(left)
                left += 1
            if right & 1:
                right -= 1
                ans.append(right)
            left >>= 1
            right >>= 1
        return ans


class PointSetRangeGcd:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover = [initial] * (4 * n)
        return

    @classmethod
    def merge(cls, x, y):
        return math.gcd(x, y)

    def _push_up(self, i):
        self.cover[i] = self.merge(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

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
        s, t, i = 0, self.n - 1, 1
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

    def range_gcd(self, left, right):
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

    def range_gcd_check(self, left, right, gcd):
        stack = [(0, self.n - 1, 1)]
        ans = 0
        while stack and ans <= 1:
            s, t, i = stack.pop()
            if self.cover[i] % gcd == 0:
                continue
            if s == t:
                if left <= s <= right and self.cover[i] % gcd:
                    ans += 1
                continue
            m = s + (t - s) // 2
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans <= 1


class LazySegmentTree:
    def __init__(self, n, combine, cover_initial, merge_cover, merge_tag, tag_initial, num_to_cover):
        self.n = n
        self.combine = combine  # method of cover push_up
        self.cover_initial = cover_initial  # cover_initial value of cover
        self.merge_cover = merge_cover  # method of tag to cover
        self.merge_tag = merge_tag  # method of tag merge
        self.tag_initial = tag_initial  # cover_initial value of tag
        self.num_to_cover = num_to_cover  # cover_initial value from num to cover
        self.lazy_tag = [self.tag_initial] * (4 * self.n)
        self.cover = [self.cover_initial] * (4 * self.n)
        return

    def _make_tag(self, i, s, t, val):
        self.cover[i] = self.merge_cover(self.cover[i], val, t - s + 1)  # cover val length
        self.lazy_tag[i] = self.merge_tag(val, self.lazy_tag[i])
        return

    def _push_up(self, i):
        self.cover[i] = self.combine(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def _push_down(self, i, s, m, t):
        if self.lazy_tag[i] != self.tag_initial:
            self._make_tag(i << 1, s, m, self.lazy_tag[i])
            self._make_tag((i << 1) | 1, m + 1, t, self.lazy_tag[i])
            self.lazy_tag[i] = self.tag_initial
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

    def range_update(self, left, right, val):

        stack = [(0, self.n - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= 0:
                if left <= s and t <= right:
                    self._make_tag(i, s, t, val)
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

    def range_query(self, left, right):
        if left == right:
            s, t, i = 0, self.n - 1, 1
            ans = self.cover_initial
            while True:
                if left <= s <= t <= right:
                    ans = self.combine(ans, self.cover[i])
                    break
                m = s + (t - s) // 2
                self._push_down(i, s, m, t)
                if left <= m:
                    s, t, i = s, m, i << 1
                if right > m:
                    s, t, i = m + 1, t, (i << 1) | 1
            return ans

        stack = [(0, self.n - 1, 1)]
        ans = self.cover_initial
        while stack:
            s, t, i = stack.pop()
            if left <= s and t <= right:
                ans = self.combine(ans, self.cover[i])
                continue
            m = s + (t - s) // 2
            self._push_down(i, s, m, t)
            if left <= m:
                stack.append((s, m, i << 1))
            if right > m:
                stack.append((m + 1, t, (i << 1) | 1))
        return ans
class SortedList:
    def __init__(self, iterable=None, _load=200):
        """Initialize sorted list instance."""
        if iterable is None:
            iterable = []
        values = sorted(iterable)
        self._len = _len = len(values)
        self._load = _load
        self._lists = _lists = [values[i:i + _load]
                                for i in range(0, _len, _load)]
        self._list_lens = [len(_list) for _list in _lists]
        self._min_s = [_list[0] for _list in _lists]
        self._fen_tree = []
        self._rebuild = True

    def _fen_build(self):
        """Build a fenwick tree instance."""
        self._fen_tree[:] = self._list_lens
        _fen_tree = self._fen_tree
        for i in range(len(_fen_tree)):
            if i | i + 1 < len(_fen_tree):
                _fen_tree[i | i + 1] += _fen_tree[i]
        self._rebuild = False

    def _fen_update(self, index, value):
        """Update `fen_tree[index] += value`."""
        if not self._rebuild:
            _fen_tree = self._fen_tree
            while index < len(_fen_tree):
                _fen_tree[index] += value
                index |= index + 1

    def _fen_query(self, end):
        """Return `sum(_fen_tree[:end])`."""
        if self._rebuild:
            self._fen_build()

        _fen_tree = self._fen_tree
        x = 0
        while end:
            x += _fen_tree[end - 1]
            end &= end - 1
        return x

    def _fen_findkth(self, k):
        """Return a pair of (the largest `idx` such that `sum(_fen_tree[:idx]) <= k`, `k - sum(_fen_tree[:idx])`)."""
        _list_lens = self._list_lens
        if k < _list_lens[0]:
            return 0, k
        if k >= self._len - _list_lens[-1]:
            return len(_list_lens) - 1, k + _list_lens[-1] - self._len
        if self._rebuild:
            self._fen_build()

        _fen_tree = self._fen_tree
        idx = -1
        for d in reversed(range(len(_fen_tree).bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < len(_fen_tree) and k >= _fen_tree[right_idx]:
                idx = right_idx
                k -= _fen_tree[idx]
        return idx + 1, k

    def _delete(self, pos, idx):
        """Delete value at the given `(pos, idx)`."""
        _lists = self._lists
        _mins = self._min_s
        _list_lens = self._list_lens

        self._len -= 1
        self._fen_update(pos, -1)
        del _lists[pos][idx]
        _list_lens[pos] -= 1

        if _list_lens[pos]:
            _mins[pos] = _lists[pos][0]
        else:
            del _lists[pos]
            del _list_lens[pos]
            del _mins[pos]
            self._rebuild = True

    def _loc_left(self, value):
        """Return an index pair that corresponds to the first position of `value` in the sorted list."""
        if not self._len:
            return 0, 0

        _lists = self._lists
        _mins = self._min_s

        lo, pos = -1, len(_lists) - 1
        while lo + 1 < pos:
            mi = (lo + pos) >> 1
            if value <= _mins[mi]:
                pos = mi
            else:
                lo = mi

        if pos and value <= _lists[pos - 1][-1]:
            pos -= 1

        _list = _lists[pos]
        lo, idx = -1, len(_list)
        while lo + 1 < idx:
            mi = (lo + idx) >> 1
            if value <= _list[mi]:
                idx = mi
            else:
                lo = mi

        return pos, idx

    def _loc_right(self, value):
        """Return an index pair that corresponds to the last position of `value` in the sorted list."""
        if not self._len:
            return 0, 0

        _lists = self._lists
        _mins = self._min_s

        pos, hi = 0, len(_lists)
        while pos + 1 < hi:
            mi = (pos + hi) >> 1
            if value < _mins[mi]:
                hi = mi
            else:
                pos = mi

        _list = _lists[pos]
        lo, idx = -1, len(_list)
        while lo + 1 < idx:
            mi = (lo + idx) >> 1
            if value < _list[mi]:
                idx = mi
            else:
                lo = mi

        return pos, idx

    def add(self, value):
        """Add `value` to sorted list."""
        _load = self._load
        _lists = self._lists
        _mins = self._min_s
        _list_lens = self._list_lens

        self._len += 1
        if _lists:
            pos, idx = self._loc_right(value)
            self._fen_update(pos, 1)
            _list = _lists[pos]
            _list.insert(idx, value)
            _list_lens[pos] += 1
            _mins[pos] = _list[0]
            if _load + _load < len(_list):
                _lists.insert(pos + 1, _list[_load:])
                _list_lens.insert(pos + 1, len(_list) - _load)
                _mins.insert(pos + 1, _list[_load])
                _list_lens[pos] = _load
                del _list[_load:]
                self._rebuild = True
        else:
            _lists.append([value])
            _mins.append(value)
            _list_lens.append(1)
            self._rebuild = True

    def discard(self, value):
        """Remove `value` from sorted list if it is a member."""
        _lists = self._lists
        if _lists:
            pos, idx = self._loc_right(value)
            if idx and _lists[pos][idx - 1] == value:
                self._delete(pos, idx - 1)

    def remove(self, value):
        """Remove `value` from sorted list; `value` must be a member."""
        _len = self._len
        self.discard(value)
        if _len == self._len:
            raise ValueError('{0!r} not in list'.format(value))

    def pop(self, index=-1):
        """Remove and return value at `index` in sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        value = self._lists[pos][idx]
        self._delete(pos, idx)
        return value

    def bisect_left(self, value):
        """Return the first index to insert `value` in the sorted list."""
        pos, idx = self._loc_left(value)
        return self._fen_query(pos) + idx

    def bisect_right(self, value):
        """Return the last index to insert `value` in the sorted list."""
        pos, idx = self._loc_right(value)
        return self._fen_query(pos) + idx

    def count(self, value):
        """Return number of occurrences of `value` in the sorted list."""
        return self.bisect_right(value) - self.bisect_left(value)

    def __len__(self):
        """Return the size of the sorted list."""
        return self._len

    def __getitem__(self, index):
        """Lookup value at `index` in sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        return self._lists[pos][idx]

    def __delitem__(self, index):
        """Remove value at `index` from sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        self._delete(pos, idx)

    def __contains__(self, value):
        """Return true if `value` is an element of the sorted list."""
        _lists = self._lists
        if _lists:
            pos, idx = self._loc_left(value)
            return idx < len(_lists[pos]) and _lists[pos][idx] == value
        return False

    def __iter__(self):
        """Return an iterator over the sorted list."""
        return (value for _list in self._lists for value in _list)

    def __reversed__(self):
        """Return a reverse iterator over the sorted list."""
        return (value for _list in reversed(self._lists)
                for value in reversed(_list))

    def __repr__(self):
        """Return strings representation of sorted list."""
        return 'SortedList({0})'.format(list(self))


class TopKSum:
    def __init__(self, k):
        self.k = k
        self.lst = SortedList()
        self.top_k_sum = 0
        return

    def add(self, num):
        self.lst.add(num)
        ind = self.lst.bisect_left(num)
        if ind <= self.k - 1:
            self.top_k_sum += num
            if len(self.lst) >= self.k + 1:
                self.top_k_sum -= self.lst[self.k]
        return

    def discard(self, num):
        ind = self.lst.bisect_left(num)
        self.lst.discard(num)
        if ind <= self.k - 1:
            self.top_k_sum -= num
            if len(self.lst) >= self.k:
                self.top_k_sum += self.lst[self.k - 1]
        return


class TopKSumSpecial:
    def __init__(self, k, bit):
        self.k = k
        self.bit = bit
        self.mask = (1 << bit) - 1
        self.lst = SortedList()
        self.top_k_sum = 0
        return

    def add(self, num):
        self.lst.add(num)
        ind = self.lst.bisect_left(num)
        if ind <= self.k - 1:
            self.top_k_sum += ((-num) >> self.bit) * ((-num) & self.mask)
            if len(self.lst) >= self.k + 1:
                num = self.lst[self.k]
                self.top_k_sum -= ((-num) >> self.bit) * ((-num) & self.mask)
        return

    def discard(self, num):
        ind = self.lst.bisect_left(num)
        self.lst.discard(num)
        if ind <= self.k - 1:
            self.top_k_sum -= ((-num) >> self.bit) * ((-num) & self.mask)
            if len(self.lst) >= self.k:
                num = self.lst[self.k - 1]
                self.top_k_sum += ((-num) >> self.bit) * ((-num) & self.mask)
        returnimport math
from functools import reduce
from math import lcm, gcd
from operator import or_, and_


class SparseTable:
    def __init__(self, lst, fun):
        """static range queries can be performed as long as the range_merge_to_disjoint fun satisfies monotonicity"""
        n = len(lst)
        self.bit = [0] * (n + 1)
        self.fun = fun
        self.n = n
        for i in range(2, n + 1):
            self.bit[i] = self.bit[i >> 1] + 1
        for i in range(n+1):
            assert self.bit[i] == (i.bit_length() - 1 if i else i.bit_length())
        self.st = [[0] * n for _ in range(self.bit[-1] + 1)]
        self.st[0] = lst
        for i in range(1, self.bit[-1] + 1):
            for j in range(n - (1 << i) + 1):
                self.st[i][j] = fun(self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))])

    def query(self, left, right):
        """index start from 0"""
        assert 0 <= left <= right < self.n
        pos = self.bit[right - left + 1]
        return self.fun(self.st[pos][left], self.st[pos][right - (1 << pos) + 1])

    def bisect_right(self, left, val, initial):
        """index start from 0"""
        assert 0 <= left < self.n
        # find the max right such that st.query(left, right) >= val
        pos = left
        pre = initial  # 0 or (1<<32)-1
        for x in range(self.bit[-1], -1, -1):
            if pos + (1 << x) - 1 < self.n and self.fun(self.st[x][pos], pre) >= val: # can by any of >= > <= <
                pre = self.fun(self.st[x][pos], pre)
                pos += (1 << x)
        # may be pos=left and st.query(left, left) < val
        if pos > left:
            pos -= 1
        else:
            pre = self.st[0][left]
        assert left <= pos < self.n
        return pos, pre

    def bisect_right_length(self, left):
        """index start from 0"""
        assert 0 <= left < self.n
        # find the max right such that st.query(left, right) < right-left+1
        pos = left
        pre = 0
        for x in range(self.bit[-1], -1, -1):
            if pos + (1 << x) - 1 < self.n and self.fun(self.st[x][pos], pre) > pos + (1 << x) - left:
                pre = self.fun(self.st[x][pos], pre)
                pos += (1 << x)
        if pos == left and self.st[0][pos] == 1:
            return True, pos
        if pos < self.n and self.fun(pre, self.st[0][pos]) == pos + 1 - left:
            return True, pos
        return False, pos


class SparseTableIndex:
    def __init__(self, lst, fun):
        """static range queries can be performed as long as the range_merge_to_disjoint fun satisfies monotonicity"""
        n = len(lst)
        self.bit = [0] * (n + 1)
        self.n = n
        self.fun = fun
        l, r, v = 1, 2, 0
        while True:
            for i in range(l, r):
                if i >= len(self.bit):
                    break
                self.bit[i] = v
            else:
                l *= 2
                r *= 2
                v += 1
                continue
            break
        self.st = [[0] * n for _ in range(self.bit[-1] + 1)]
        self.st[0] = list(range(n))
        for i in range(1, self.bit[-1] + 1):
            for j in range(n - (1 << i) + 1):
                a, b = self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))]
                if self.fun(lst[a], lst[b]) == lst[a]:
                    self.st[i][j] = a
                else:
                    self.st[i][j] = b
        self.lst = lst
        return

    def query(self, left, right):
        """index start from 0"""
        assert 0 <= left <= right <= self.n - 1
        pos = self.bit[right - left + 1]
        a, b = self.st[pos][left], self.st[pos][right - (1 << pos) + 1]
        if self.fun(self.lst[a], self.lst[b]) == self.lst[a]:
            return a
        return b


class SparseTable2D:
    def __init__(self, matrix, method="max"):
        m, n = len(matrix), len(matrix[0])
        a, b = int(math.log2(m)) + 1, int(math.log2(n)) + 1

        if method == "max":
            self.fun = max
        elif method == "min":
            self.fun = min
        elif method == "gcd":
            self.fun = self.gcd
        elif method == "lcm":
            self.fun = min
        elif method == "or":
            self.fun = self._or
        else:
            self.fun = self._and

        self.dp = [[[[0 for _ in range(b)] for _ in range(a)] for _ in range(1000)] for _ in range(1000)]

        for i in range(a):
            for j in range(b):
                for x in range(m - (1 << i) + 1):
                    for y in range(n - (1 << j) + 1):
                        if i == 0 and j == 0:
                            self.dp[x][y][i][j] = matrix[x][y]
                        elif i == 0:
                            self.dp[x][y][i][j] = self.fun([self.dp[x][y][i][j - 1],
                                                            self.dp[x][y + (1 << (j - 1))][i][j - 1]])
                        elif j == 0:
                            self.dp[x][y][i][j] = self.fun([self.dp[x][y][i - 1][j],
                                                            self.dp[x + (1 << (i - 1))][y][i - 1][j]])
                        else:
                            self.dp[x][y][i][j] = self.fun([self.dp[x][y][i - 1][j - 1],
                                                            self.dp[x + (1 << (i - 1))][y][i - 1][j - 1],
                                                            self.dp[x][y + (1 << (j - 1))][i - 1][j - 1],
                                                            self.dp[x + (1 << (i - 1))][y + (1 << (j - 1))][i - 1][
                                                                j - 1]])
        return

    @staticmethod
    def max(args):
        return reduce(max, args)

    @staticmethod
    def min(args):
        return reduce(min, args)

    @staticmethod
    def gcd(args):
        return reduce(gcd, args)

    @staticmethod
    def lcm(args):
        return reduce(lcm, args)

    @staticmethod
    def _or(args):
        return reduce(or_, args)

    @staticmethod
    def _and(args):
        return reduce(and_, args)

    def query(self, x, y, x1, y1):
        # index start from 0 and left up corner is (x, y) and right down corner is (x1, y1)
        k = int(math.log2(x1 - x + 1))
        p = int(math.log2(y1 - y + 1))
        ans = self.fun([self.dp[x][y][k][p],
                        self.dp[x1 - (1 << k) + 1][y][k][p],
                        self.dp[x][y1 - (1 << p) + 1][k][p],
                        self.dp[x1 - (1 << k) + 1][y1 - (1 << p) + 1][k][p]])
        return ans
class BlockSize:
    def __init__(self):
        return

    @staticmethod
    def get_divisor_split(n):
        # Decompose the interval [1, n] into each interval whose divisor of n does not exceed the range
        if n == 1:
            return [1], [[1, 1]]
        m = int(n ** 0.5)
        pre = []
        post = []
        for x in range(1, m + 1):
            pre.append(x)
            post.append(n // x)
        if pre[-1] == post[-1]:
            post.pop()
        post.reverse()
        res = pre + post

        cnt = [res[0]] + [res[i + 1] - res[i] for i in range(len(res) - 1)]
        k = len(cnt)
        assert k == 2 * m - int(m == n // m)

        right = [n // (k - i) for i in range(1, k)]
        pre = n // k
        seg = [[1, pre - 1]] if pre > 1 else []
        for num in right:
            seg.append([pre, num])
            pre = num + 1
        assert sum([ls[1] - ls[0] + 1 for ls in seg]) == n
        return cnt, seg



class PointXorRangeXor:
    def __init__(self, n: int, initial=0) -> None:
        self.n = n
        self.t = [initial] * (self.n + 1)
        return

    def _lowest_bit(self, i: int) -> int:
        assert 1 <= i <= self.n
        return i & (-i)

    def _pre_xor(self, i: int) -> int:
        assert 0 <= i < self.n
        i += 1
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
        nums = [self._pre_xor(i) for i in range(self.n)]
        for i in range(self.n - 1, 0, -1):
            nums[i] -= nums[i - 1]
        return nums

    def point_xor(self, i: int, val: int) -> None:
        assert 0 <= i < self.n
        i += 1
        while i < len(self.t):
            self.t[i] ^= val
            i += self._lowest_bit(i)
        return

    def range_xor(self, x: int, y: int) -> int:
        assert 0 <= x <= y < self.n
        res = self._pre_xor(y) ^ self._pre_xor(x - 1) if x else self._pre_xor(y)
        return res


class PointAddRangeSum:
    def __init__(self, n: int, initial=0) -> None:
        """index from 1 to n"""
        self.n = n
        self.t = [initial] * (self.n + 1)  # default nums = [0]*n
        return

    def _lowest_bit(self, i: int) -> int:
        assert 1 <= i <= self.n
        return i & (-i)

    def _pre_sum(self, i: int) -> int:
        """index start from 1 and the prefix sum of nums[:i] which is 0-index"""
        assert 0 <= i < self.n
        i += 1
        val = 0
        while i:
            val += self.t[i]
            i -= self._lowest_bit(i)
        return val

    def build(self, nums) -> None:
        """initialize the tree array"""
        assert len(nums) == self.n
        pre = [0] * (self.n + 1)
        for i in range(self.n):
            pre[i + 1] = pre[i] + nums[i]
            # meaning of self.t[i+1]
            self.t[i + 1] = pre[i + 1] - pre[i + 1 - self._lowest_bit(i + 1)]
        return

    def get(self):
        """get the original nums sometimes for debug"""
        nums = [self._pre_sum(i) for i in range(self.n)]
        for i in range(self.n - 1, 0, -1):
            nums[i] -= nums[i - 1]
        return nums

    def point_add(self, i: int, val: int) -> None:
        """index start from 1 and the value val can be any inter including positive and negative number"""
        assert 0 <= i < self.n
        i += 1
        while i < len(self.t):
            self.t[i] += val
            i += self._lowest_bit(i)
        return

    def range_sum(self, x: int, y: int) -> int:
        assert 0 <= x <= y < self.n
        """0-index"""
        res = self._pre_sum(y) - self._pre_sum(x - 1) if x else self._pre_sum(y)
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
        assert 0 <= x <= self.n
        return x


class PointChangeRangeSum:
    def __init__(self, n: int) -> None:
        # index from 1 to n
        self.n = n
        self.t = [0] * (self.n + 1)  # default nums = [0]*n
        return

    def _lowest_bit(self, i: int) -> int:
        assert 1 <= i <= self.n
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

class PointAscendPreMax:
    def __init__(self, n, initial=-math.inf):
        self.n = n
        self.initial = initial
        self.t = [initial] * (n + 1)

    @staticmethod
    def _lowest_bit(i):
        return i & (-i)

    def pre_max(self, i):
        assert 0 <= i <= self.n - 1  # max(nums[:i+1])
        i += 1
        mx = self.initial
        while i:
            mx = mx if mx > self.t[i] else self.t[i]
            i -= self._lowest_bit(i)
        return mx

    def point_ascend(self, i, mx):
        assert 0 <= i <= self.n - 1
        i += 1
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] > mx else mx
            i += self._lowest_bit(i)
        return

class PointAscendPreMaxIndex:
    def __init__(self, n, initial=-math.inf):
        self.n = n
        self.initial = initial
        self.t = [initial] * (n + 1)
        self.ind = [-1] * (n + 1)

    @staticmethod
    def _lowest_bit(i):
        return i & (-i)

    def pre_max(self, i):
        assert 0 <= i <= self.n - 1  # max(nums[:i+1])
        i += 1
        mx = self.initial
        res = -1
        while i:
            if self.t[i] > mx:
                mx = self.t[i]
                res = self.ind[i]
            i -= self._lowest_bit(i)
        return mx, res

    def point_ascend(self, i, mx, index):
        assert 0 <= i <= self.n - 1
        i += 1
        while i < len(self.t):
            if self.t[i] < mx:
                self.t[i] = mx
                self.ind[i] = index
            i += self._lowest_bit(i)
        return


class PointAscendPostMax:
    def __init__(self, n, initial=-math.inf):
        self.n = n
        self.initial = initial
        self.t = [initial] * (n + 1)

    @staticmethod
    def _lowest_bit(i):
        return i & (-i)

    def post_max(self, i):
        assert 0 <= i <= self.n - 1  # max(nums[i:])
        i = self.n - i - 1
        i += 1
        mx = self.initial
        while i:
            mx = mx if mx > self.t[i] else self.t[i]
            i -= self._lowest_bit(i)
        return mx

    def point_ascend(self, i, mx):
        assert 0 <= i <= self.n - 1
        i = self.n - i - 1
        i += 1
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] > mx else mx
            i += self._lowest_bit(i)
        return


class PointAscendRangeMax:
    def __init__(self, n: int, initial=-math.inf) -> None:
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
    def __init__(self, n, initial=math.inf):
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
        assert 0 <= i <= self.n - 1  # # min(nums[:i+1])
        i += 1
        val = self.initial
        while i:
            val = val if val < self.t[i] else self.t[i]
            i -= self._lowest_bit(i)
        return val

    def point_descend(self, i, val):
        assert 0 <= i <= self.n - 1
        i += 1
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] < val else val
            i += self._lowest_bit(i)
        return


class PointDescendPostMin:
    def __init__(self, n, initial=math.inf):
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

    def post_min(self, i):
        assert 0 <= i <= self.n - 1
        i = self.n - 1 - i  # min(nums[i:])
        i += 1
        val = self.initial
        while i:
            val = val if val < self.t[i] else self.t[i]
            i -= self._lowest_bit(i)
        return val

    def point_descend(self, i, val):
        assert 0 <= i <= self.n - 1
        i = self.n - 1 - i
        i += 1
        while i < len(self.t):
            self.t[i] = self.t[i] if self.t[i] < val else val
            i += self._lowest_bit(i)
        return


class PointDescendRangeMin:
    def __init__(self, n: int, initial=math.inf) -> None:
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
        self.tree_floor = [[float('math.inf')] * (n + 1) for _ in range(m + 1)]  # point keep descend
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
        max_val = math.inf
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


class BinaryTrieXorDict:
    def __init__(self, bit_length):
        self.dct = dict()
        self.bit_length = bit_length
        return

    def add(self, num, cnt):
        cur = self.dct
        for i in range(self.bit_length, -1, -1):
            cur["cnt"] = cur.get("cnt", 0) + cnt
            w = 1 if num & (1 << i) else 0
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        cur["cnt"] = cur.get("cnt", 0) + cnt
        return

    def get_cnt_smaller_xor(self, num, ceil):

        def dfs(xor, cur, i):
            nonlocal res
            if xor > ceil:
                return
            if i == -1:
                res += cur["cnt"]
                return
            if xor + (1 << (i + 2) - 1) <= ceil:
                res += cur["cnt"]
                return
            w = 1 if num & (1 << i) else 0
            if 1 - w in cur:
                dfs(xor | (1 << i), cur[1 - w], i - 1)
            if w in cur:
                dfs(xor, cur[w], i - 1)
            return

        res = 0
        dfs(0, self.dct, self.bit_length)
        return res


class BinaryTrieXor:
    def __init__(self, max_num, num_cnt):  # bitwise xor
        if max_num <= 0:
            max_num = 1
        if num_cnt <= 0:
            num_cnt = 1
        binary_state = 2
        self.max_bit = max_num.bit_length() - 1
        self.cnt_bit = num_cnt.bit_length()
        self.node_cnt = (self.max_bit + 1) * num_cnt * binary_state
        self.son_and_cnt = [0] * (self.node_cnt + 1)
        self.ind = 1
        self.mask = (1 << self.cnt_bit) - 1

    def initial(self):
        for i in range(self.node_cnt + 1):
            self.son_and_cnt[i] = 0
        self.ind = 1

    def add(self, num: int, c=1) -> bool:
        cur = 0
        self.son_and_cnt[cur] += c
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            if not self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit:
                self.son_and_cnt[(cur << 1) | bit] |= (self.ind << self.cnt_bit)
                self.ind += 1
            cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            self.son_and_cnt[cur] += c
        return True

    def remove(self, num: int, c=1) -> bool:
        if self.son_and_cnt[0] & self.mask < c:
            return False
        cur = 0
        self.son_and_cnt[0] -= c
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            if cur == 0 or self.son_and_cnt[cur] & self.mask < c:
                return False
            self.son_and_cnt[cur] -= c
        return True

    def count(self, num: int):
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            if cur == 0 or self.son_and_cnt[cur] & self.mask == 0:
                return 0
        return self.son_and_cnt[cur] & self.mask

    def get_maximum_xor(self, x: int) -> int:
        """get maximum result for constant x ^ element in array"""
        if self.son_and_cnt[0] & self.mask == 0:
            return -math.inf
        res = 0
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            nxt = self.son_and_cnt[(cur << 1) | (bit ^ 1)] >> self.cnt_bit
            if nxt == 0 or self.son_and_cnt[nxt] & self.mask == 0:
                cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            else:
                res |= 1 << k
                cur = nxt
        return res

    def get_minimum_xor(self, x: int) -> int:
        """get minimum result for constant x ^ element in array"""
        if self.son_and_cnt[0] & self.mask == 0:
            return math.inf
        res = 0
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            nxt = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            if nxt == 0 or self.son_and_cnt[nxt] & self.mask == 0:
                res |= 1 << k
                cur = self.son_and_cnt[(cur << 1) | (bit ^ 1)] >> self.cnt_bit
            else:
                cur = nxt
        return res

    def get_kth_maximum_xor(self, x: int, rk) -> int:
        """get kth maximum result for constant x ^ element in array"""
        assert rk >= 1
        if self.son_and_cnt[0] & self.mask < rk:
            return -math.inf
        res = 0
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            nxt = self.son_and_cnt[(cur << 1) | (bit ^ 1)] >> self.cnt_bit
            if nxt == 0 or self.son_and_cnt[nxt] & self.mask < rk:
                if nxt:
                    rk -= self.son_and_cnt[nxt] & self.mask
                cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            else:
                res |= 1 << k
                cur = nxt
        return res

    def get_cnt_smaller_xor(self, x: int, y: int) -> int:
        """get cnt result for constant x ^ element <= y in array"""
        if self.son_and_cnt[0] & self.mask == 0:
            return 0
        res = 0
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            if not (y >> k) & 1:
                nxt = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
                if nxt == 0 or self.son_and_cnt[nxt] & self.mask == 0:
                    return res
                cur = nxt
            else:
                nxt = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
                if nxt:
                    res += self.son_and_cnt[nxt] & self.mask
                nxt = self.son_and_cnt[(cur << 1) | (bit ^ 1)] >> self.cnt_bit
                if nxt == 0 or self.son_and_cnt[nxt] & self.mask == 0:
                    return res
                cur = nxt
        res += self.son_and_cnt[cur] & self.mask
        return res


class BinaryTrieXorLimited:
    def __init__(self, max_num, num_cnt):  # bitwise xor
        if max_num <= 0:
            max_num = 1
        if num_cnt <= 0:
            num_cnt = 1
        binary_state = 2
        self.max_bit = max_num.bit_length() - 1
        self.cnt_bit = num_cnt.bit_length()
        self.node_cnt = (self.max_bit + 1) * num_cnt * binary_state
        self.son_and_cnt = [0] * (self.node_cnt + 1)
        self.floor = [math.inf] * (self.node_cnt + 1)
        self.ind = 1
        self.mask = (1 << self.cnt_bit) - 1

    def initial(self):
        for i in range(self.node_cnt + 1):
            self.son_and_cnt[i] = 0
        self.ind = 1

    def add(self, num: int, c=1) -> bool:
        cur = 0
        self.son_and_cnt[cur] += c
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            if not self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit:
                self.son_and_cnt[(cur << 1) | bit] |= (self.ind << self.cnt_bit)
                self.ind += 1
            cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
            if num < self.floor[cur]:
                self.floor[cur] = num
            self.son_and_cnt[cur] += c
        return True

    def get_maximum_xor_limited(self, x: int, m) -> int:
        """get maximum result for constant x ^ element in array and element <= m"""
        if self.son_and_cnt[0] & self.mask == 0:
            return -1
        res = 0
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            nxt = self.son_and_cnt[(cur << 1) | (bit ^ 1)] >> self.cnt_bit
            if nxt == 0 or self.son_and_cnt[nxt] & self.mask == 0 or self.floor[nxt] > m:
                cur = self.son_and_cnt[(cur << 1) | bit] >> self.cnt_bit
                if cur == 0 or self.son_and_cnt[cur] & self.mask == 0 or self.floor[cur] > m:
                    return -1
            else:
                res |= 1 << k
                cur = nxt
        return res


class StringTrieSearch:
    def __init__(self, most_word, word_cnt, string_state=26):  # search index
        assert most_word >= 1
        assert word_cnt >= 1
        self.string_state = string_state
        self.cnt_bit = word_cnt.bit_length()
        self.node_cnt = most_word * self.string_state
        self.son_and_ind = [0] * (self.node_cnt + 1)
        self.ind = 0
        self.mask = (1 << self.cnt_bit) - 1

    def initial(self):
        for i in range(self.node_cnt + 1):
            self.son_and_ind[i] = 0
        self.ind = 0

    def add(self, word, ind):
        assert 1 <= ind <= word_cnt
        cur = 0  # word: List[int]
        for bit in word:
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
        self.son_and_ind[cur] |= ind
        return

    def search(self, word):
        res = []
        cur = 0
        for bit in word:
            cur = self.son_and_ind[bit + self.string_state * cur] >> self.cnt_bit
            if not cur:
                break
            if self.son_and_ind[cur] & self.mask:
                res.append(self.son_and_ind[cur] & self.mask)
        return res

    def add_ind(self, word, ind):
        assert 1 <= ind <= word_cnt
        cur = 0  # word: List[int]
        for bit in word:
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
            if not self.son_and_ind[cur] & self.mask:
                self.son_and_ind[cur] |= ind
        return

    def search_ind(self, word):
        res = cur = 0
        for bit in word:
            cur = self.son_and_ind[bit + self.string_state * cur] >> self.cnt_bit
            if not cur:
                break
            if self.son_and_ind[cur] & self.mask:
                res = self.son_and_ind[cur] & self.mask
        return res

    def search_length(self, word):
        cur = res = 0
        for bit in word:
            cur = self.son_and_ind[bit + self.string_state * cur] >> self.cnt_bit
            if not cur:
                break
            res += 1
        return res

    def add_cnt(self, word, ind):
        cur = res = 0
        for bit in word:
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
            if self.son_and_ind[cur] & self.mask:
                res += 1
        self.son_and_ind[cur] |= ind
        res += 1
        return res

    def add_exist(self, word, ind):
        cur = 0
        res = False
        for bit in word:
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                res = True
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
            if self.son_and_ind[cur] & self.mask:
                res += 1
        self.son_and_ind[cur] |= ind
        return res

    def add_bin(self, word, ind):
        cur = 0
        for w in word:
            bit = int(w)
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
        self.son_and_ind[cur] |= ind
        return

    def add_int(self, word, ind):
        cur = 0
        for bit in word:
            if not self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_ind[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_ind[bit + cur * self.string_state] >> self.cnt_bit
        self.son_and_ind[cur] |= ind
        return

    def search_for_one_difference(self, word):
        n = len(word)
        stack = [(0, 0, 0)]
        while stack:
            cur, i, c = stack.pop()
            if i == n:
                return True
            bit = word[i]
            if not c:
                for bit2 in range(26):
                    if bit2 != bit:
                        nex = self.son_and_ind[bit2 + self.string_state * cur] >> self.cnt_bit
                        if nex:
                            stack.append((nex, i + 1, c + 1))
            cur = self.son_and_ind[bit + self.string_state * cur] >> self.cnt_bit
            if cur:
                stack.append((cur, i + 1, c))
        return False


class StringTriePrefix:
    def __init__(self, most_word, word_cnt, string_state=26):  # prefix count
        assert most_word >= 1
        assert word_cnt >= 1
        self.string_state = string_state
        self.cnt_bit = word_cnt.bit_length()
        self.node_cnt = most_word * self.string_state
        self.son_and_cnt = [0] * (self.node_cnt + 1)
        self.ind = 0
        self.mask = (1 << self.cnt_bit) - 1

    def initial(self):
        for i in range(self.node_cnt + 1):
            self.son_and_cnt[i] = 0
        self.ind = 0

    def add(self, word, val=1):
        cur = 0  # word: List[int]
        self.son_and_cnt[cur] += val
        for bit in word:
            if not self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_cnt[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit
            self.son_and_cnt[cur] += val
        return

    def count(self, word):
        res = cur = 0  # word: List[int]
        for bit in word:
            cur = self.son_and_cnt[bit + self.string_state * cur] >> self.cnt_bit
            if not cur or self.son_and_cnt[cur] & self.mask == 0:
                break
            res += self.son_and_cnt[cur] & self.mask
        return res

    def count_end(self, word):
        cur = 0  # word: List[int]
        for bit in word:
            cur = self.son_and_cnt[bit + self.string_state * cur] >> self.cnt_bit
            if not cur or self.son_and_cnt[cur] & self.mask == 0:
                return 0
        return self.son_and_cnt[cur] & self.mask

    def add_end(self, word, val=1):
        cur = 0  # word: List[int]
        for bit in word:
            if not self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit:
                self.ind += 1
                self.son_and_cnt[bit + cur * self.string_state] |= self.ind << self.cnt_bit
            cur = self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit
        self.son_and_cnt[cur] += val
        return

    def count_pre_end(self, word):
        res = cur = 0  # word: List[int]
        for bit in word:
            cur = self.son_and_cnt[bit + self.string_state * cur] >> self.cnt_bit
            if not cur:
                break
            res += self.son_and_cnt[cur] & self.mask
        return res

from operator import add


class PointSetAddRangeSum:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover = [initial] * 2 * self.n
        return

    @staticmethod
    def combine(a, b):
        return a + b

    def push_up(self, i):
        self.cover[i] = self.combine(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def build(self, nums):
        for i in range(self.n):
            self.cover[i + self.n] = nums[i]
        for i in range(self.n - 1, 0, -1):
            self.push_up(i)
        return

    def get(self):
        return self.cover[self.n:]

    def point_set(self, ind, val):
        ind += self.n
        self.cover[ind] = val
        while ind > 1:
            ind //= 2
            self.push_up(ind)
        return

    def point_add(self, ind, val):
        ind += self.n
        self.cover[ind] += val
        while ind > 1:
            ind //= 2
            self.push_up(ind)
        return

    def range_sum(self, left, right):
        ans_left = ans_right = 0
        left += self.n
        right += self.n + 1
        while left < right:
            if left & 1:
                ans_left = self.combine(ans_left, self.cover[left])
                left += 1
            if right & 1:
                right -= 1
                ans_right = self.combine(self.cover[right], ans_right)
            left >>= 1
            right >>= 1
        return self.combine(ans_left, ans_right)


class RangeAddPointGet:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (2 * self.n)
        return

    def build(self, nums):
        for i in range(self.n):
            self.cover[i + self.n] = nums[i]
        return

    def push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        return

    def range_add(self, left, right, val):
        left += self.n
        right += self.n + 1

        while left < right:
            if left & 1:
                self.cover[left] += val
                left += 1
            if right & 1:
                right -= 1
                self.cover[right] += val
            left >>= 1
            right >>= 1
        return

    def get(self):
        for i in range(1, self.n):
            self.cover[i << 1] += self.cover[i]
            self.cover[(i << 1) | 1] += self.cover[i]
            self.cover[i] = 0
        return self.cover[self.n:]

    def point_get(self, ind):
        ans = 0
        ind += self.n
        while ind > 0:
            ans += self.cover[ind]
            ind //= 2
        return ans


class LazySegmentTree:
    def __init__(self, n, combine, cover_initial, merge_cover, merge_tag, tag_initial, num_to_cover):
        self.n = n
        self.combine = combine  # method of cover push_up
        self.cover_initial = cover_initial  # cover_initial value of cover
        self.merge_cover = merge_cover  # method of tag to cover
        self.merge_tag = merge_tag  # method of tag merge
        self.tag_initial = tag_initial  # cover_initial value of tag
        self.num_to_cover = num_to_cover  # cover_initial value from num to cover
        self.lazy_tag = [self.tag_initial] * (2 * self.n)
        self.h = 0
        while (1 << self.h) < n:
            self.h += 1
        self.cover = [self.cover_initial] * (2 * self.n)
        self.cnt = [1] * (2 * self.n)
        for i in range(self.n - 1, 0, -1):
            self.cnt[i] = self.cnt[i << 1] + self.cnt[(i << 1) | 1]
        return

    def build(self, nums):
        for i in range(self.n):
            self.cover[i + self.n] = self.num_to_cover(nums[i])
        for i in range(self.n - 1, 0, -1):
            self.cover[i] = self.combine(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def push_up(self, i):
        while i > 1:
            i >>= 1
            self.cover[i] = self.combine(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def push_down(self, i):
        for s in range(self.h, 0, -1):
            x = i >> s
            if self.lazy_tag[x] != self.tag_initial:
                self.make_tag(x << 1, self.lazy_tag[x])
                self.make_tag((x << 1) | 1, self.lazy_tag[x])
                self.lazy_tag[x] = self.tag_initial
        x = i
        if self.lazy_tag[x] != self.tag_initial:
            if (i << 1) < self.n * 2:
                self.make_tag(x << 1, self.lazy_tag[x])
            if ((i << 1) | 1) < self.n * 2:
                self.make_tag((x << 1) | 1, self.lazy_tag[x])
            self.lazy_tag[x] = self.tag_initial
        return

    def make_tag(self, i, val):
        self.cover[i] = self.merge_cover(self.cover[i], val, self.cnt[i])  # cover val length
        self.lazy_tag[i] = self.merge_tag(val, self.lazy_tag[i])
        return

    def range_update(self, left, right, val):
        left += self.n
        right += self.n + 1
        ll = left
        rr = right
        self.push_down(ll)
        self.push_down(rr - 1)
        while left < right:
            if left & 1:
                self.make_tag(left, val)
                left += 1
            if right & 1:
                right -= 1
                self.make_tag(right, val)
            left >>= 1
            right >>= 1
        self.push_down(ll)
        self.push_down(rr - 1)
        self.push_up(ll)
        self.push_up(rr - 1)
        return

    def get(self):
        for i in range(1, self.n):
            if self.lazy_tag[i] != self.tag_initial:
                self.make_tag(i << 1, self.lazy_tag[i])
                self.make_tag((i << 1) | 1, self.lazy_tag[i])
                self.lazy_tag[i] = self.tag_initial
        return self.cover[self.n:]

    def point_get(self, ind):
        ans = 0
        ind += self.n
        while ind > 0:
            ans += self.lazy_tag[ind]
            ind //= 2
        return ans

    def range_query(self, left, right):
        ans_left = ans_right = self.cover_initial
        left += self.n
        right += self.n + 1
        self.push_down(left)
        self.push_down(right - 1)
        while left < right:
            if left & 1:
                ans_left = self.combine(ans_left, self.cover[left])
                left += 1
            if right & 1:
                right -= 1
                ans_right = self.combine(self.cover[right], ans_right)
            left >>= 1
            right >>= 1
        return self.combine(ans_left, ans_right)


class LazySegmentTreeLength:
    def __init__(self, n, combine_cover, cover_initial, merge_cover_tag, merge_tag_tag, tag_initial):
        self.n = n
        self.cover_initial = cover_initial
        self.merge_cover_tag = merge_cover_tag
        self.merge_tag_tag = merge_tag_tag
        self.tag_initial = tag_initial
        self.lazy_tag = [self.tag_initial] * (2 * self.n)
        self.h = 0
        while (1 << self.h) < n:
            self.h += 1
        self.combine_cover = combine_cover
        self.cover = [self.cover_initial] * (2 * self.n)
        return

    def build(self, nums):
        for i in range(self.n):
            if nums[i]:
                self.cover[i + self.n] = (1, 1, 0, 1, 1, 1)
            else:
                self.cover[i + self.n] = (-1, -1, 1, 0, 1, 0)
        for i in range(self.n - 1, 0, -1):
            self.cover[i] = self.combine_cover(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def push_up(self, i):
        while i > 1:
            i >>= 1
            self.cover[i] = self.combine_cover(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def push_down(self, i):
        for s in range(self.h, 0, -1):
            x = i >> s
            if self.lazy_tag[x] != self.tag_initial:
                self.make_tag(x << 1, self.lazy_tag[x])
                self.make_tag((x << 1) | 1, self.lazy_tag[x])
                self.lazy_tag[x] = self.tag_initial
        x = i
        if self.lazy_tag[x] != self.tag_initial:
            if (i << 1) < self.n * 2:
                self.make_tag(x << 1, self.lazy_tag[x])
            if ((i << 1) | 1) < self.n * 2:
                self.make_tag((x << 1) | 1, self.lazy_tag[x])
            self.lazy_tag[x] = self.tag_initial
        return

    def make_tag(self, i, val):
        self.cover[i] = self.merge_cover_tag(self.cover[i], val)
        self.lazy_tag[i] = self.merge_tag_tag(val, self.lazy_tag[i])
        return

    def range_update(self, left, right, val):
        left += self.n
        right += self.n + 1
        ll = left
        rr = right
        self.push_down(ll)
        self.push_down(rr - 1)
        while left < right:
            if left & 1:
                self.make_tag(left, val)
                left += 1
            if right & 1:
                right -= 1
                self.make_tag(right, val)
            left >>= 1
            right >>= 1
        self.push_down(ll)
        self.push_down(rr - 1)

        self.push_up(ll)
        self.push_up(rr - 1)
        return

    def get(self):
        for i in range(1, self.n):
            if self.lazy_tag[i] != self.tag_initial:
                self.make_tag(i << 1, self.lazy_tag[i])
                self.make_tag((i << 1) | 1, self.lazy_tag[i])
                self.lazy_tag[i] = self.tag_initial
        return self.cover[self.n:]

    def point_get(self, ind):
        ans = 0
        ind += self.n
        while ind > 0:
            ans += self.lazy_tag[ind]
            ind //= 2
        return ans

    def range_query(self, left, right):
        ans_left = ans_right = self.cover_initial
        left += self.n
        right += self.n + 1
        self.push_down(left)
        self.push_down(right - 1)
        while left < right:
            if left & 1:
                ans_left = self.combine_cover(ans_left, self.cover[left])
                left += 1
            if right & 1:
                right -= 1
                ans_right = self.combine_cover(self.cover[right], ans_right)
            left >>= 1
            right >>= 1
        return self.combine_cover(ans_left, ans_right)
import math
from collections import defaultdict




class BagDP:
    def __init__(self):
        return

    @staticmethod
    def bin_split_1(num):
        # binary optimization refers to continuous operations such as 1.2.4.x
        # instead of binary 10101 corresponding to 1
        if not num:
            return []
        lst = []
        x = 1
        while x <= num:
            lst.append(x)
            num -= x
            x *= 2
        if num:
            lst.append(num)
        return lst

    @staticmethod
    def bin_split_2(num):
        # split from large to small to ensure that there are no identical positive numbers other than 1
        if not num:
            return []
        lst = []
        while num:
            lst.append((num + 1) // 2)
            num //= 2
        lst.reverse()
        return lst

    @staticmethod
    def one_dimension_limited(n, nums):
        # 01 backpack
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for i in range(n, num - 1, -1):
                dp[i] += dp[i - num]
        return dp[n]

    @staticmethod
    def one_dimension_unlimited(n, nums):
        # complete backpack
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for i in range(num, n + 1):
                dp[i] += dp[i - num]
        return dp[n]

    @staticmethod
    def two_dimension_limited(m, n, nums):
        # 2D 01 backpack
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        for a, b in nums:
            for i in range(m, a - 1, -1):
                for j in range(n, b - 1, -1):
                    dp[i][j] += dp[i - a][j - b]
        return dp[m][n]

    @staticmethod
    def two_dimension_unlimited(m, n, nums):
        # 2D complete backpack
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        for a, b in nums:
            for i in range(a, m + 1):
                for j in range(b, n + 1):
                    dp[i][j] += dp[i - a][j - b]
        return dp[m][n]

    def continuous_bag_with_bin_split(self, n, nums):
        # Continuous 01 Backpack Using Binary Optimization
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for x in self.bin_split_1(num):
                for i in range(n, x - 1, -1):
                    dp[i] += dp[i - x]
        return dp[n]

    @staticmethod
    def group_bag_limited(n, d, nums):
        # group backpack
        pre = [math.inf] * (n + 1)
        pre[0] = 0
        for r, z in nums:
            cur = pre[:]  # The key is that we need group backpacks here
            for x in range(1, z + 1):
                cost = d + x * r
                for i in range(n, x - 1, -1):
                    if pre[i - x] + cost < cur[i]:
                        cur[i] = pre[i - x] + cost
            pre = cur[:]
        if pre[n] < math.inf:
            return pre[n]
        return -1

    @staticmethod
    def group_bag_unlimited(nums):
        # Calculate the number of solutions for decomposing n into the sum of squares of four numbers
        n = max(nums)
        dp = [[0] * 5 for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, int(math.sqrt(n)) + 1):
            x = i * i
            for j in range(x, n + 1):
                for k in range(1, 5):
                    if dp[j - x][k - 1]:
                        dp[j][k] += dp[j - x][k - 1]
        return [sum(dp[num]) for num in nums]

    @staticmethod
    def one_dimension_limited_use_dct(nums):
        # One dimensional finite backpack
        # using a dictionary for transfer records with negative numbers
        pre = defaultdict(lambda: -math.inf)
        # can also use [0]*2*s where s is the sum(abs(x) for x in nums)
        pre[0] = 0
        for s, f in nums:
            cur = pre.copy()
            for p in pre:
                cur[p + s] = max(cur[p + s], pre[p] + f)
            pre = cur
        ans = 0
        for p in pre:
            if p >= 0 and pre[p] >= 0:
                ans = ans if ans > p + pre[p] else p + pre[p]
        return ans
from functools import lru_cache
from itertools import accumulate


class DigitalDP:
    def __init__(self):
        return

    @staticmethod
    def count_bin(n):
        # calculate the number of occurrences of positive integer binary bit 1 from 1 to n

        @lru_cache(None)
        def dfs(i, is_limit, is_num, cnt):
            if i == m:
                if is_num:
                    return cnt
                return 0
            res = 0
            if not is_num:
                res += dfs(i + 1, False, False, cnt)
            low = 0 if is_num else 1
            high = int(st[i]) if is_limit else 1
            for x in range(low, high + 1):
                res += dfs(i + 1, is_limit and high == x, True, cnt + int(i == w) * x)
            return res

        st = bin(n)[2:]
        m = len(st)
        ans = []  # From binary high to binary low
        for w in range(m):
            cur = dfs(0, True, False, 0)
            ans.append(cur)
            dfs.cache_clear()
        return ans

    @staticmethod
    def count_bin2(m):
        cnt = []
        val = 1
        while val <= m:
            # a, b 分别是循环节和剩余元素个数
            a, b = divmod(m, val * 2)
            # 统计 1 的数量
            cnt.append(a * val + min(max(b - val + 1, 0), val))
            val *= 2
        return cnt


    @staticmethod
    def count_digit_1(n):
        k, kk = 0, 1
        ans = 0
        while n >= kk:
            ans += (n // (kk * 10)) * kk + min(max(n % (kk * 10) - kk + 1, 0), kk)
            k += 1
            kk *= 10
        return ans

    @staticmethod
    def count_digit_dfs(num, d):
        # Calculate the number of occurrences of digit d within 1 to num

        @lru_cache(None)
        def dfs(i, cnt, is_limit, is_num):
            if i == n:
                if is_num:
                    return cnt
                return 0
            res = 0
            if not is_num:
                res += dfs(i + 1, 0, False, False)

            floor = 0 if is_num else 1
            ceil = int(s[i]) if is_limit else 9
            for x in range(floor, ceil + 1):
                res += dfs(i + 1, cnt + int(x == d), is_limit and ceil == x, True)
            return res

        s = str(num)
        n = len(s)
        return dfs(0, 0, True, False)

    @staticmethod
    def count_digit_dp(num, d):
        # Calculate the number of occurrences of digit d within 1 to num by iteration
        num += 1
        lst = [int(x) for x in str(num)]
        n = len(lst)
        pre = [0] * (n + 1)
        c = 0
        zero = 0
        cnt = 1
        for i in range(n):
            cur = [0] * (n + 1)
            for x in range(i + 1):
                cur[x] += 9 * pre[x]
                cur[x + 1] += pre[x]
            for w in range(lst[i]):
                cur[c + int(w == d)] += 1
            c += int(lst[i] == d)
            pre = cur
            zero += cnt
            cnt *= 10
        ans = sum(pre[i] * i for i in range(n + 1))
        return ans if d else ans - zero

    @staticmethod
    def count_digit_sum(num):
        # Calculate the number of occurrences of digit d within 1 to num
        num += 1
        lst = [int(x) for x in str(num)]
        n = len(lst)
        pre_sum = 0
        dp = [0] * (9 * n + 1)
        for i in range(n):
            ndp = [0] * (9 * n + 1)
            pre_dp = list(accumulate(dp, initial=0))
            for digit_sum in range(9 * n + 1):
                ndp[digit_sum] += pre_dp[digit_sum + 1] - pre_dp[max(0, digit_sum - 9)]
            for cur_digit in range(lst[i]):
                ndp[pre_sum + cur_digit] += 1
            pre_sum += lst[i]
            dp = ndp
        ans = sum(dp[x] * x for x in range(9 * n + 1))
        return ans

    @staticmethod
    def count_num_base(num, d):
        # Use decimal to calculate the number of digits from 1 to num without the digit d
        assert 1 <= d <= 9  # If 0 is not included, use digital DP for calculation
        s = str(num)
        i = s.find(str(d))
        if i != -1:
            if d:
                s = s[:i] + str(d - 1) + (len(s) - i - 1) * "9"
            else:
                s = s[:i - 1] + str(int(s[i - 1]) - 1) + (len(s) - i - 1) * "9"
            num = int(s)

        lst = []
        while num:
            lst.append(num % 10)
            if d and lst[-1] >= d:
                lst[-1] -= 1
            elif not d and lst[-1] == 0:
                num *= 10
                num -= 1
                lst.append(num % 10)
            num //= 10
        lst.reverse()

        ans = 0
        for x in lst:
            ans *= 9
            ans += x
        return ans

    @staticmethod
    def count_num_dp(num, d):

        # Use decimal to calculate the number of digits from 1 to num without the digit d
        assert 0 <= d <= 9

        @lru_cache(None)
        def dfs(i: int, is_limit: bool, is_num: bool) -> int:
            if i == m:
                return int(is_num)

            res = 0
            if not is_num:
                res = dfs(i + 1, False, False)
            up = int(s[i]) if is_limit else 9
            for x in range(0 if is_num else 1, up + 1):
                if x != d:
                    res += dfs(i + 1, is_limit and x == up, True)
            return res

        s = str(num)
        m = len(s)
        return dfs(0, True, False)

    @staticmethod
    def get_kth_without_d(k, d):
        # Use decimal to calculate the k-th digit without digit d 0<=d<=9
        assert 0 <= d <= 9
        lst = []
        st = list(range(10))
        st.remove(d)
        while k:
            if d:
                lst.append(k % 9)
                k //= 9
            else:
                lst.append((k - 1) % 9)
                k = (k - 1) // 9
        lst.reverse()
        # It can also be solved using binary search and digit DP
        ans = [str(st[i]) for i in lst]
        return int("".join(ans))
class DateTime:
    def __init__(self):
        self.leap_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.not_leap_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return

    def is_leap_year(self, yy):
        # Determine whether it is a leap year
        assert sum(self.leap_month) == 366
        assert sum(self.not_leap_month) == 365
        return yy % 400 == 0 or (yy % 4 == 0 and yy % 100 != 0)

    def year_month_day_cnt(self, yy, mm):
        ans = self.leap_month[mm - 1] if self.is_leap_year(yy) else self.not_leap_month[mm - 1]
        return ans

    def is_valid(self, yy, mm, dd):
        if not [1900, 1, 1] <= [yy, mm, dd] <= [2006, 11, 4]:
            return False
        day = self.year_month_day_cnt(yy, mm)
        if not 1 <= dd <= day:
            return False
        return True
class LinearDP:
    def __init__(self):
        return

    @staticmethod
    def liner_dp_template(nums):
        # example of lis（longest increasing sequence）
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = 1
            for j in range(i):
                if nums[i] > nums[j] and dp[j] + 1 > dp[i + 1]:
                    dp[i + 1] = dp[j] + 1
        return max(dp)



class MatrixDP:
    def __init__(self):
        return

    @staticmethod
    def lcp(s, t):
        # longest common prefix of s[i:] and t[j:]
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == t[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
        return dp

    @staticmethod
    def min_distance(word1: str, word2: str):
        m, n = len(word1), len(word2)
        dp = [[math.inf] * (n + 1) for _ in range(m + 1)]
        # edit distance
        for i in range(m + 1):
            dp[i][n] = m - i
        for j in range(n + 1):
            dp[m][j] = n - j
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j + 1] + 1,
                               dp[i + 1][j + 1] + int(word1[i] != word2[j]))
        return dp[0][0]

    @staticmethod
    def path_mul_mod(m, n, k, grid):
        # calculate the modulus of the product of the matrix from the upper left corner to the lower right corner
        dp = [[set() for _ in range(n)] for _ in range(m)]
        dp[0][0].add(grid[0][0] % k)
        for i in range(1, m):
            x = grid[i][0]
            for p in dp[i - 1][0]:
                dp[i][0].add((p * x) % k)
        for j in range(1, n):
            x = grid[0][j]
            for p in dp[0][j - 1]:
                dp[0][j].add((p * x) % k)

        for i in range(1, m):
            for j in range(1, n):
                x = grid[i][j]
                for p in dp[i][j - 1]:
                    dp[i][j].add((p * x) % k)
                for p in dp[i - 1][j]:
                    dp[i][j].add((p * x) % k)
        ans = sorted(list(dp[-1][-1]))
        return ans

    @staticmethod
    def maximal_square(matrix) -> int:

        # The maximum square sub matrix with all value equal to 1
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        ans = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    dp[i + 1][j + 1] = min(dp[i][j], dp[i + 1][j], dp[i][j + 1]) + 1
                    if dp[i + 1][j + 1] > ans:
                        ans = dp[i + 1][j + 1]
        # the ans is side length and ans**2 is area
        return ans ** 2

    @staticmethod
    def longest_common_sequence(s1, s2, s3) -> str:
        # Longest common subsequence LCS can be extended to 3D and 4D or higher dimension
        m, n, k = len(s1), len(s2), len(s3)
        # length of lcs
        dp = [[[0] * (k + 1) for _ in range(n + 1)] for _ in range(m + 1)]
        # example of lcs
        res = [[[""] * (k + 1) for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                for p in range(k):
                    if s1[i] == s2[j] == s3[p]:
                        if dp[i + 1][j + 1][p + 1] < dp[i][j][p] + 1:
                            dp[i + 1][j + 1][p + 1] = dp[i][j][p] + 1
                            res[i + 1][j + 1][p + 1] = res[i][j][p] + s1[i]
                    else:
                        for a, b, c in [[1, 1, 0], [0, 1, 1], [1, 0, 1]]:  # transfer formula
                            if dp[i + 1][j + 1][p + 1] < dp[i + a][j + b][p + c]:
                                dp[i + 1][j + 1][p + 1] = dp[i + a][j + b][p + c]
                                res[i + 1][j + 1][p + 1] = res[i + a][j + b][p + c]
        return res[m][n][k]
from src.utils.fast_io import FastIO


class WeightedTree:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_weight = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.parent = [-1]
        self.order = 0
        self.start = [-1]
        self.end = [-1]
        self.parent = [-1]
        self.depth = [0]
        self.order_to_node = [-1]
        return

    def add_directed_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_weight.append(w)
        self.edge_from.append(u)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v, w)
        self.add_directed_edge(v, u, w)
        return

    def get_edge_ids(self, u):
        assert 0 <= u < self.n
        i = self.point_head[u]
        ans = []
        while i:
            ans.append(i)
            i = self.edge_next[i]
        return

    def dfs_order(self, root=0):

        self.order = 0
        # index is original node value is dfs self.order
        self.start = [-1] * self.n
        # index is original node value is the maximum subtree dfs self.order
        self.end = [-1] * self.n
        # index is original node and value is its self.parent
        self.parent = [-1] * self.n
        stack = [root]
        # self.depth of every original node
        self.depth = [0] * self.n
        # index is dfs self.order and value is original node
        self.order_to_node = [-1] * self.n
        while stack:
            i = stack.pop()
            if i >= 0:
                self.start[i] = self.order
                self.order_to_node[self.order] = i
                self.end[i] = self.order
                self.order += 1
                stack.append(~i)
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    # the self.order of son nodes can be assigned for lexicographical self.order
                    if j != self.parent[i]:
                        self.parent[j] = i
                        self.depth[j] = self.depth[i] + 1
                        stack.append(j)
                    ind = self.edge_next[ind]
            else:
                i = ~i
                if self.parent[i] != -1:
                    self.end[self.parent[i]] = self.end[i]

        return

    def heuristic_merge(self):
        ans = [0] * self.n
        sub = [None for _ in range(self.n)]
        index = list(range(self.n))
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        stack = [0]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != self.parent[i]:
                        self.parent[j] = i
                        self.depth[j] = self.depth[i] + 1
                        stack.append(j)
                    ind = self.edge_next[ind]
            else:
                i = ~i
                sub[index[i]] = {self.depth[i]: 1}
                ans[i] = self.depth[i]
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != self.parent[i]:
                        a, b = index[i], index[j]
                        if len(sub[a]) > len(sub[b]):
                            res = ans[i]
                            a, b = b, a
                        else:
                            res = ans[j]

                        for x in sub[a]:
                            sub[b][x] = sub[b].get(x, 0) + sub[a][x]
                            if (sub[b][x] > sub[b][res]) or (sub[b][x] == sub[b][res] and x < res):
                                res = x
                        sub[a] = None
                        ans[i] = res
                        index[i] = b
                    ind = self.edge_next[ind]

        return [ans[i] - self.depth[i] for i in range(self.n)]

    # class Graph(WeightedTree):
    def tree_dp(self, nums):
        ans = [0] * self.n
        parent = [-1] * self.n
        stack = [0]
        res = max(nums)
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != parent[i]:
                        parent[j] = i
                        stack.append(j)
                    ind = self.edge_next[ind]
            else:
                i = ~i
                ind = self.point_head[i]
                a = b = 0
                while ind:
                    j = self.edge_to[ind]
                    if j != parent[i]:
                        cur = ans[j] - self.edge_weight[ind]
                        if cur > a:
                            a, b = cur, a
                        elif cur > b:
                            b = cur
                    ind = self.edge_next[ind]
                res = max(res, a + b + nums[i])
                ans[i] = a + nums[i]
        return res


class ReadGraph:
    def __init__(self):
        return

    @staticmethod
    def read(ac=FastIO()):
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        stack = [0]
        sub = [0] * n
        while stack:
            val = stack.pop()
            if val >= 0:
                x, fa = val // n, val % n
                stack.append(~val)
                for y in dct[x]:
                    if y != fa:
                        stack.append(y * n + x)
            else:
                val = ~val
                x, fa = val // n, val % n
                for y in dct[x]:
                    if y != fa:
                        sub[x] += sub[y]
                sub[x] += 1
        return sub


class ReRootDP:
    def __init__(self):
        return

    @staticmethod
    def get_tree_distance_weight(dct, weight):
        # Calculate the total distance from each node of the tree to all other nodes
        # each node has weight

        n = len(dct)
        sub = weight[:]
        s = sum(weight)  # default equal to [1]*n
        ans = [0] * n  # distance to all other nodes

        # first bfs to get ans[0] and subtree weight from bottom to top
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ans[i] += ans[j] + sub[j]

        # second bfs to get all ans[i]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    # sub[j] equal j up to i
                    # s - sub[j] equal to i down to j
                    # change = -sub[j] + s - sub[j]
                    ans[j] = ans[i] - sub[j] + s - sub[j]
                    stack.append([j, i])
        return ans

    @staticmethod
    def get_tree_centroid(dct) -> int:
        # the smallest centroid of tree
        # equal the node with minimum of maximum subtree node cnt
        # equivalent to the node which has the shortest distance from all other nodes
        n = len(dct)
        sub = [1] * n  # subtree size of i-th node rooted by 0
        ma = [0] * n  # maximum subtree node cnt or i-rooted
        ma[0] = n
        center = 0
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ma[i] = ma[i] if ma[i] > sub[j] else sub[j]
                # like re-rooted dp to check the maximum subtree size
                ma[i] = ma[i] if ma[i] > n - sub[i] else n - sub[i]
                if ma[i] < ma[center] or (ma[i] == ma[center] and i < center):
                    center = i
        return center

    @staticmethod
    def get_tree_distance(dct):
        # Calculate the total distance from each node of the tree to all other nodes

        n = len(dct)
        sub = [1] * n  # Number of subtree nodes
        ans = [0] * n  # The sum of distances to all other nodes

        # first bfs to get ans[0] and subtree weight from bottom to top
        stack = [(0, -1, 1)]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append((i, fa, 0))
                for j in dct[i]:
                    if j != fa:
                        stack.append((j, i, 1))
            else:
                for j in dct[i]:
                    if j != fa:
                        sub[i] += sub[j]
                        ans[i] += ans[j] + sub[j]

        # second bfs to get all ans[i]
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            for j in dct[i]:
                if j != fa:
                    # sub[j] equal j up to i
                    # s - sub[j] equal to i down to j
                    # change = -sub[j] + s - sub[j]
                    ans[j] = ans[i] - sub[j] + n - sub[j]
                    stack.append((j, i))
        return ans

    @staticmethod
    def get_tree_distance_max(dct):
        # Calculate the maximum distance from each node of the tree to all other nodes
        # point bfs on diameter can also be used

        n = len(dct)
        sub = [[0, 0] for _ in range(n)]

        # first bfs compute the largest distance and second large distance from bottom to up
        stack = [[0, -1, 1]]
        while stack:
            i, fa, state = stack.pop()
            if state:
                stack.append([i, fa, 0])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i, 1])
            else:
                a, b = sub[i]
                for j in dct[i]:
                    if j != fa:
                        x = sub[j][0] + 1
                        if x >= a:
                            a, b = x, a
                        elif x >= b:
                            b = x
                sub[i] = [a, b]

        # second bfs compute large distance from up to bottom
        stack = [(0, -1, 0)]
        ans = [s[0] for s in sub]
        while stack:
            i, fa, d = stack.pop()
            ans[i] = ans[i] if ans[i] > d else d
            for j in dct[i]:
                if j != fa:
                    nex = d
                    x = sub[j][0] + 1
                    a, b = sub[i]
                    # distance from current child nodes excluded
                    if x == a:
                        nex = nex if nex > b else b
                    else:
                        nex = nex if nex > a else a
                    stack.append((j, i, nex + 1))
        return ans

    @staticmethod
    def get_tree_distance_max_weighted(dct, weights):
        # Calculate the maximum distance from each node of the tree to all other nodes
        # point bfs on diameter can also be used

        n = len(dct)
        sub = [[0, 0] for _ in range(n)]

        # first bfs compute the largest distance and second large distance from bottom to up
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        stack.append((j, i))
            else:
                i = ~i
                a, b = sub[i]
                for j in dct[i]:
                    if j != fa:
                        x = sub[j][0] + weights[j]
                        if x >= a:
                            a, b = x, a
                        elif x >= b:
                            b = x
                sub[i] = [a, b]

        # second bfs compute large distance from up to bottom
        stack = [(0, -1, 0)]
        ans = [s[0] for s in sub]
        while stack:
            i, fa, d = stack.pop()
            ans[i] = ans[i] if ans[i] > d else d
            for j in dct[i]:
                if j != fa:
                    nex = d
                    x = sub[j][0] + weights[j]
                    a, b = sub[i]
                    # distance from current child nodes excluded
                    if x == a:
                        nex = nex if nex > b else b
                    else:
                        nex = nex if nex > a else a
                    stack.append((j, i, nex + weights[i]))
        return ansfrom src.graph.union_find.template import UnionFind


class BinarySearchTree:

    def __init__(self):
        return

    @staticmethod
    def build_with_unionfind(nums):
        """build binary search tree by the order of nums with unionfind"""

        n = len(nums)
        ind = list(range(n))
        ind.sort(key=lambda it: nums[it])
        rank = {idx: i for i, idx in enumerate(ind)}

        dct = [[] for _ in range(n)]
        uf = UnionFind(n)
        post = {}
        for i in range(n - 1, -1, -1):
            x = rank[i]
            if x + 1 in post:
                r = uf.find(post[x + 1])
                dct[i].append(r)
                uf.union_left(i, r)
            if x - 1 in post:
                r = uf.find(post[x - 1])
                dct[i].append(r)
                uf.union_left(i, r)
            post[x] = i
        return dct

    @staticmethod
    def build_with_stack(nums):
        """build binary search tree by the order of nums with stack"""

        n = len(nums)

        lst = sorted(nums)
        dct = {num: i + 1 for i, num in enumerate(lst)}
        ind = {num: i for i, num in enumerate(nums)}

        order = [dct[i] for i in nums]
        father, occur, stack = [0] * (n + 1), [0] * (n + 1), []
        deep = [0] * (n + 1)
        for i, x in enumerate(order, 1):
            occur[x] = i

        for x, i in enumerate(occur):
            while stack and occur[stack[-1]] > i:
                if occur[father[stack[-1]]] < i:
                    father[stack[-1]] = x
                stack.pop()
            if stack:
                father[x] = stack[-1]
            stack.append(x)

        for x in order:
            deep[x] = 1 + deep[father[x]]

        dct = [[] for _ in range(n)]
        for i in range(1, n + 1):
            if father[i]:
                u, v = father[i] - 1, i - 1
                x, y = ind[lst[u]], ind[lst[v]]
                dct[x].append(y)
        return dct
class BipartiteMatching:
    def __init__(self, n, m):
        self._n = n
        self._m = m
        self._to = [[] for _ in range(n)]

    def add_edge(self, a, b):
        self._to[a].append(b)

    def solve(self):
        n, m, to = self._n, self._m, self._to
        prev = [-1] * n
        root = [-1] * n
        p = [-1] * n
        q = [-1] * m
        updated = True
        while updated:
            updated = False
            s = []
            s_front = 0
            for i in range(n):
                if p[i] == -1:
                    root[i] = i
                    s.append(i)
            while s_front < len(s):
                v = s[s_front]
                s_front += 1
                if p[root[v]] != -1:
                    continue
                for u in to[v]:
                    if q[u] == -1:
                        while u != -1:
                            q[u] = v
                            p[v], u = u, p[v]
                            v = prev[v]
                        updated = True
                        break
                    u = q[u]
                    if prev[u] != -1:
                        continue
                    prev[u] = v
                    root[u] = root[v]
                    s.append(u)
            if updated:
                for i in range(n):
                    prev[i] = -1
                    root[i] = -1
        return [(v, p[v]) for v in range(n) if p[v] != -1]


class Hungarian:
    def __init__(self):
        # Bipartite graph maximum math without weight
        return

    @staticmethod
    def dfs_recursion(n, m, dct):
        assert len(dct) == m

        def hungarian(i):
            for j in dct[i]:
                if not visit[j]:
                    visit[j] = True
                    if match[j] == -1 or hungarian(match[j]):
                        match[j] = i
                        return True
            return False

        # left group size is n
        match = [-1] * n
        ans = 0
        for x in range(m):
            # right group size is m
            visit = [False] * n
            if hungarian(x):
                ans += 1
        return ans

    @staticmethod
    def bfs_iteration(n, m, dct):

        assert len(dct) == m

        match = [-1] * n
        ans = 0
        for i in range(m):
            hungarian = [0] * m
            visit = [0] * n
            stack = [[i, 0]]
            while stack:
                x, ind = stack[-1]
                if ind == len(dct[x]) or hungarian[x]:
                    stack.pop()
                    continue
                y = dct[x][ind]
                if not visit[y]:
                    visit[y] = 1
                    if match[y] == -1:
                        match[y] = x
                        hungarian[x] = 1
                    else:
                        stack.append([match[y], 0])
                else:
                    if hungarian[match[y]]:
                        match[y] = x
                        hungarian[x] = 1
                    stack[-1][1] += 1
            if hungarian[i]:
                ans += 1
        return ans
import math
from collections import defaultdict, deque
from heapq import heappush, heappop

from src.data_structure.sorted_list.template import SortedList


class WeightedGraphForShortestPathMST:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_weight = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.dis = [math.inf]
        self.edge_id = 1
        return

    def add_directed_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_weight.append(w)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v, w)
        self.add_directed_edge(v, u, w)
        return

    def get_edge_ids(self, u):
        assert 0 <= u < self.n
        i = self.point_head[u]
        ans = []
        while i:
            ans.append(i)
            i = self.edge_next[i]
        return

    def dijkstra(self, src=0, initial=0):
        self.dis = [math.inf] * (self.n + 1)
        stack = [initial * self.n + src]
        self.dis[src] = initial
        while stack:
            val = heappop(stack)
            d, u = val // self.n, val % self.n
            if self.dis[u] < d:
                continue
            i = self.point_head[u]
            while i:
                w = self.edge_weight[i]
                j = self.edge_to[i]
                dj = d + w
                if dj < self.dis[j]:
                    self.dis[j] = dj
                    heappush(stack, dj * self.n + j)
                i = self.edge_next[i]
        return

    def shortest_path_mst(self, root=0, ceil=0):
        dis = [math.inf] * self.n
        stack = [root]
        dis[root] = 0
        edge_ids = [-1] * self.n
        weights = [math.inf] * self.n
        while stack:
            val = heappop(stack)
            d, u = val // self.n, val % self.n
            if dis[u] < d:
                continue
            ind = self.point_head[u]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = w + d
                if dj < dis[j] or (dj == dis[j] and w < weights[j]):
                    dis[j] = dj
                    edge_ids[j] = (ind + 1) // 2
                    weights[j] = w
                    heappush(stack, dj * self.n + j)
                ind = self.edge_next[ind]
        dis = [dis[i] * self.n + i for i in range(self.n)]
        dis.sort()
        return [edge_ids[x % self.n] for x in dis[1:ceil + 1]]

class LimitedWeightedGraph:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_weight1 = [0]
        self.edge_weight2 = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.time = [math.inf]
        self.edge_id = 1
        return

    def add_directed_edge(self, u, v, t, c):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_weight1.append(t)
        self.edge_weight2.append(c)
        self.edge_from.append(u)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v, t, c):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v, t, c)
        self.add_directed_edge(v, u, t, c)
        return

    def get_edge_ids(self, u):
        assert 0 <= u < self.n
        i = self.point_head[u]
        ans = []
        while i:
            ans.append(i)
            i = self.edge_next[i]
        return

    def limited_dijkstra(self, src=0, des=0, ceil=math.inf, initial=0):
        self.time = [ceil] * (self.n + 1)
        stack = [initial * ceil * self.n + 0 * self.n + src]
        self.time[src] = initial  # min(cost) when time<ceil
        while stack:
            val = heappop(stack)  # cost time point
            cost, tm, u = val // (ceil * self.n), (val % (ceil * self.n)) // self.n, (val % (ceil * self.n)) % self.n
            if u == des:
                return cost
            i = self.point_head[u]
            while i:
                t = self.edge_weight1[i]
                c = self.edge_weight2[i]
                j = self.edge_to[i]
                dj = tm + t
                if dj < self.time[j]:
                    self.time[j] = dj
                    heappush(stack, (cost + c) * ceil * self.n + dj * self.n + j)
                i = self.edge_next[i]
        return -1

    def limited_dijkstra_tuple(self, src=0, des=0, ceil=math.inf, initial=0):
        self.time = [ceil] * (self.n + 1)
        stack = [(initial, 0, src)]
        self.time[src] = initial  # min(cost) when time<ceil
        while stack:
            cost, tm, u = heappop(stack)  # cost time point
            if u == des:
                return cost
            i = self.point_head[u]
            while i:
                t = self.edge_weight1[i]
                c = self.edge_weight2[i]
                j = self.edge_to[i]
                dj = tm + t
                if dj < self.time[j]:
                    self.time[j] = dj
                    heappush(stack, (cost + c, dj, j))
                i = self.edge_next[i]
        return -1




class WeightedGraphForDijkstra:
    def __init__(self, n, inf=math.inf):
        self.n = n
        self.point_head = [0] * self.n
        self.edge_weight = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.inf = inf
        return

    def add_directed_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.edge_weight.append(w)
        self.edge_from.append(i)
        self.edge_to.append(j)
        self.edge_next.append(self.point_head[i])
        self.point_head[i] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.add_directed_edge(i, j, w)
        self.add_directed_edge(j, i, w)
        return

    def get_edge_ids(self, i):
        assert 0 <= i < self.n
        ind = self.point_head[i]
        ans = []
        while ind:
            ans.append(ind)
            ind = self.edge_next[ind]
        return

    def dijkstra_for_shortest_path(self, src=0, initial=0):
        dis = [self.inf] * self.n
        stack = [initial * self.n + src]
        dis[src] = initial
        while stack:
            val = heappop(stack)
            d, i = val // self.n, val % self.n
            if dis[i] < d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, dj * self.n + j)
                ind = self.edge_next[ind]
        return dis

    def dijkstra_for_shortest_path_float(self, src=0, initial=0):
        dis = [self.inf] * self.n
        stack = [(initial, src)]
        dis[src] = initial
        while stack:
            d, i = heappop(stack)
            if dis[i] < d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j]:
                    dis[j] = dj
                    heappush(stack, (dj, j))
                ind = self.edge_next[ind]
        return dis

    def dijkstra_for_longest_path(self, src=0, initial=0):
        dis = [-self.inf] * self.n
        stack = [-initial * self.n - src]
        dis[src] = initial
        while stack:
            val = -heappop(stack)
            d, i = val // self.n, val % self.n
            if dis[i] > d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj > dis[j]:
                    dis[j] = dj
                    heappush(stack, -dj * self.n - j)
                ind = self.edge_next[ind]
        return dis

    def dijkstra_for_cnt_of_shortest_path(self, src=0, initial=0, mod=-1):
        """number of shortest path"""
        assert 0 <= src < self.n
        dis = [self.inf] * self.n
        cnt = [0] * self.n
        dis[src] = initial
        cnt[src] = 1
        stack = [initial * self.n + src]
        while stack:
            val = heappop(stack)
            d, i = val // self.n, val % self.n
            if dis[i] < d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j]:
                    dis[j] = dj
                    cnt[j] = cnt[i]
                    heappush(stack, dj * self.n + j)
                elif dj == dis[j]:
                    cnt[j] += cnt[i]
                    if mod != -1:
                        cnt[j] %= mod
                ind = self.edge_next[ind]
        return dis, cnt

    def dijkstra_for_strictly_second_shortest_path(self, src=0, initial=0):
        dis = [self.inf] * self.n * 2
        stack = [initial * self.n + src]
        dis[src * 2] = initial
        while stack:
            val = heappop(stack)
            d, i = val // self.n, val % self.n
            if dis[i * 2 + 1] < d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j * 2]:
                    dis[j * 2 + 1] = dis[j * 2]
                    dis[j * 2] = dj
                    heappush(stack, dj * self.n + j)
                elif dis[j * 2] < dj < dis[j * 2 + 1]:
                    dis[j * 2 + 1] = dj
                    heappush(stack, dj * self.n + j)
                ind = self.edge_next[ind]
        return dis

    def dijkstra_for_cnt_of_strictly_second_shortest_path(self, src=0, initial=0, mod=-1):
        dis = [self.inf] * self.n * 2
        dis[src * 2] = initial
        cnt = [0] * self.n * 2
        cnt[src * 2] = 1
        stack = [initial * 2 * self.n + src * 2]
        while stack:
            val = heappop(stack)
            d, i = val // self.n // 2, val % (2 * self.n)
            i, state = i // 2, i % 2
            val = i * 2 + state
            if dis[val] < d:
                continue
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j * 2]:
                    dis[j * 2 + 1] = dis[j * 2]
                    cnt[j * 2 + 1] = cnt[j * 2]
                    dis[j * 2] = dj
                    cnt[j * 2] = cnt[val]
                    heappush(stack, dj * 2 * self.n + j * 2)
                    heappush(stack, dis[j * 2 + 1] * 2 * self.n + j * 2 + 1)
                elif dj == dis[j * 2]:
                    cnt[j * 2] += cnt[val]
                    if mod != -1:
                        cnt[j * 2] %= mod
                elif dj < dis[j * 2 + 1]:
                    dis[j * 2 + 1] = dj
                    cnt[j * 2 + 1] = cnt[val]
                    heappush(stack, dj * 2 * self.n + j * 2 + 1)
                elif dj == dis[j * 2 + 1]:
                    cnt[j * 2 + 1] += cnt[val]
                    if mod != -1:
                        cnt[j * 2 + 1] %= mod
                ind = self.edge_next[ind]
        return dis, cnt

    def dijkstra_for_shortest_path_from_src_to_dst(self, src, dst):
        assert 0 <= src < self.n
        assert 0 <= dst < self.n
        dis = [self.inf] * self.n
        stack = [0 * self.n + src]
        dis[src] = 0
        parent = [-1] * self.n
        while stack:
            val = heappop(stack)
            d, i = val // self.n, val % self.n
            if dis[i] < d:
                continue
            if dst == i:
                break
            ind = self.point_head[i]
            while ind:
                w = self.edge_weight[ind]
                j = self.edge_to[ind]
                dj = d + w
                if dj < dis[j]:
                    dis[j] = dj
                    parent[j] = i
                    heappush(stack, dj * self.n + j)
                ind = self.edge_next[ind]
        path = [dst]
        while parent[path[-1]] != -1:
            path.append(parent[path[-1]])
        path.reverse()
        return path, dis[dst]

    def bfs_for_shortest_path_from_src_to_dst(self, src, dst):
        dis = [self.inf] * self.n
        dis[src] = 0
        stack = [src]
        parent = [-1] * self.n
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if self.edge_weight[ind] == 1:
                        dj = dis[i] + 1
                        if dj < dis[j]:
                            dis[j] = dj
                            parent[j] = i
                            nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex
        path = [dst]
        while parent[path[-1]] != -1:
            path.append(parent[path[-1]])
        path.reverse()
        return path, dis[dst]

    def bfs_for_shortest_path(self, src=0, initial=0):
        dis = [self.inf] * self.n
        dis[src] = initial
        stack = [src]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if self.edge_weight[ind] == 1:
                        dj = dis[i] + 1
                        if dj < dis[j]:
                            dis[j] = dj
                            nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex
        return dis

    def bfs_for_shortest_path_with_odd_and_even(self, src=0, initial=0):
        dis = [self.inf] * self.n * 2
        dis[src * 2] = initial
        stack = [src * 2]
        while stack:
            nex = []
            for val in stack:
                i = val // 2
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if self.edge_weight[ind] == 1:
                        dj = dis[val] + 1
                        sj = j * 2 + dj % 2
                        if dj < dis[sj]:
                            dis[sj] = dj
                            nex.append(sj)
                    ind = self.edge_next[ind]
            stack = nex
        return dis

    def bfs_for_cnt_of_shortest_path(self, src=0, initial=0, mod=-1):
        dis = [self.inf] * self.n
        dis[src] = initial
        stack = [src]
        cnt = [0] * self.n
        cnt[src] = 1
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if self.edge_weight[ind] == 1:
                        dj = dis[i] + 1
                        if dj < dis[j]:
                            dis[j] = dj
                            nex.append(j)
                            cnt[j] = cnt[i]
                        elif dj == dis[j]:
                            cnt[j] += cnt[i]
                            if mod != -1:
                                cnt[j] %= mod
                    ind = self.edge_next[ind]
            stack = nex
        return dis, cnt

    def bfs_for_cnt_of_strictly_second_shortest_path(self, src=0, initial=0, mod=-1):
        dis = [self.inf] * self.n * 2
        dis[src * 2] = initial
        cnt = [0] * self.n * 2
        cnt[src * 2] = 1
        stack = [src * 2]
        while stack:
            nex = []
            for val in stack:
                i = val // 2
                ind = self.point_head[i]
                d = dis[val]
                while ind:
                    w = self.edge_weight[ind]
                    if w == 1:
                        j = self.edge_to[ind]
                        dj = d + 1
                        if dj < dis[j * 2]:
                            dis[j * 2 + 1] = dis[j * 2]
                            cnt[j * 2 + 1] = cnt[j * 2]
                            nex.append(j * 2)
                            dis[j * 2] = dj
                            cnt[j * 2] = cnt[val]
                        elif dj == dis[j * 2]:
                            cnt[j * 2] += cnt[val]
                            if mod != -1:
                                cnt[j * 2] %= mod
                        elif dj < dis[j * 2 + 1]:
                            dis[j * 2 + 1] = dj
                            cnt[j * 2 + 1] = cnt[val]
                            nex.append(j * 2 + 1)
                        elif dj == dis[j * 2 + 1]:
                            cnt[j * 2 + 1] += cnt[val]
                            if mod != -1:
                                cnt[j * 2 + 1] %= mod
                    ind = self.edge_next[ind]
            stack = nex
        return dis, cnt


class UnDirectedShortestCycle:
    def __init__(self):
        return

    @staticmethod
    def find_shortest_cycle_with_node(n: int, dct) -> int:
        # brute force by point
        ans = math.inf
        for i in range(n):
            dist = [math.inf] * n
            par = [-1] * n
            dist[i] = 0
            stack = [(0, i)]
            while stack:
                _, x = heappop(stack)
                for child in dct[x]:
                    if dist[x] > ans:
                        break
                    if dist[child] > dct[x][child] + dist[x]:
                        dist[child] = dct[x][child] + dist[x]
                        par[child] = x
                        heappush(stack, (dist[child], child))
                    elif par[x] != child and par[child] != x:
                        cur = dist[x] + dist[child] + dct[x][child]
                        ans = ans if ans < cur else cur
        return ans if ans != math.inf else -1

    @staticmethod
    def find_shortest_cycle_with_edge(n: int, dct, edges) -> int:
        # brute force by edge

        ans = math.inf
        for x, y, w in edges:
            dct[x].pop(y)
            dct[y].pop(x)

            dis = [math.inf] * n
            stack = [(0, x)]
            dis[x] = 0

            while stack:
                d, i = heappop(stack)
                if dis[i] < d:
                    continue
                if i == y:
                    break
                for j in dct[i]:
                    dj = dct[i][j] + d
                    if dj < dis[j]:
                        dis[j] = dj
                        heappush(stack, (dj, j))

            ans = ans if ans < dis[y] + w else dis[y] + w
            dct[x][y] = w
            dct[y][x] = w
        return ans if ans < math.inf else -1
from collections import deque


class DirectedEulerPath:
    def __init__(self, n, pairs):
        self.n = n
        # directed edge
        self.pairs = pairs
        # edges order on euler path
        self.paths = list()
        # nodes order on euler path
        self.nodes = list()
        self.exist = False
        self.get_euler_path()
        return

    def get_euler_path(self):
        # in and out degree sum of node
        degree = [0] * self.n
        edge = [[] for _ in range(self.n)]
        for i, j in self.pairs:
            degree[i] += 1
            degree[j] -= 1
            edge[i].append(j)

        # visited by lexicographical order
        for i in range(self.n):
            edge[i].sort(reverse=True)  # which can be adjusted

        # find the start point and end point of euler path
        starts = []
        ends = []
        zero = 0
        for i in range(self.n):
            if degree[i] == 1:
                starts.append(i)  # start node which out_degree - in_degree = 1
            elif degree[i] == -1:
                ends.append(i)  # start node which out_degree - in_degree = -1
            else:
                zero += 1  # other nodes have out_degree - in_degree = 0
        del degree

        if not len(starts) == len(ends) == 1:
            if zero != self.n:
                return
            starts = [0]

        # Hierholzer algorithm with iterative implementation
        stack = [starts[0]]
        while stack:
            current = stack[-1]
            if edge[current]:
                next_node = edge[current].pop()
                stack.append(next_node)
            else:
                self.nodes.append(current)
                if len(stack) > 1:
                    self.paths.append([stack[-2], current])
                stack.pop()
        self.paths.reverse()
        self.nodes.reverse()

        # Pay attention to determining which edge passes through before calculating the Euler path
        if len(self.nodes) == len(self.pairs) + 1:
            self.exist = True
        return


class UnDirectedEulerPath:
    def __init__(self, n, pairs):
        self.n = n
        # undirected edge
        self.pairs = pairs
        self.paths = list()
        self.nodes = list()
        self.exist = False
        self.get_euler_path()
        return

    def get_euler_path(self):
        degree = [0] * self.n
        edge = [dict() for _ in range(self.n)]
        for i, j in self.pairs:
            degree[i] += 1
            degree[j] += 1
            edge[i][j] = edge[i].get(j, 0) + 1
            edge[j][i] = edge[j].get(i, 0) + 1
        edge_dct = [deque(sorted(dt)) for dt in edge]  # visited by order of node id
        starts = []
        zero = 0
        for i in range(self.n):
            if degree[i] % 2:  # which can be start point or end point
                starts.append(i)
            else:
                zero += 1
        del degree

        if not len(starts) == 2:
            # just two nodes have odd degree and others have even degree
            if zero != self.n:
                return
            starts = [0]

        # Hierholzer algorithm with iterative implementation
        stack = [starts[0]]
        while stack:
            current = stack[-1]
            next_node = None
            while edge_dct[current]:
                if not edge[current][edge_dct[current][0]]:
                    edge_dct[current].popleft()
                    continue
                nex = edge_dct[current][0]
                if edge[current][nex]:
                    edge[current][nex] -= 1
                    edge[nex][current] -= 1
                    next_node = nex
                    stack.append(next_node)
                    break
            if next_node is None:
                self.nodes.append(current)
                if len(stack) > 1:
                    pre = stack[-2]
                    self.paths.append([pre, current])
                stack.pop()
        self.paths.reverse()
        self.nodes.reverse()
        # Pay attention to determining which edge passes through before calculating the Euler path
        if len(self.nodes) == len(self.pairs) + 1:
            self.exist = True
        return
import math


class WeightedGraphForFloyd:
    def __init__(self, n, inf=math.inf):
        self.n = n
        self.inf = inf
        self.dis = [self.inf] * self.n * self.n
        for i in range(self.n):
            self.dis[i * self.n + i] = 0
        self.cnt = []
        return

    def add_undirected_edge_initial(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.dis[i * self.n + j] = self.dis[j * self.n + i] = min(self.dis[i * self.n + j], w)
        return

    def add_directed_edge_initial(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.dis[i * self.n + j] = min(self.dis[i * self.n + j], w)
        return

    def initialize_undirected(self):
        for k in range(self.n):
            self.update_point_undirected(k)
        return

    def initialize_directed(self):
        for k in range(self.n):
            self.update_point_directed(k)
        return

    def add_undirected_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        for x in range(self.n):
            if self.dis[x * self.n + i] == self.inf:
                continue
            for y in range(x + 1, self.n):
                cur = min(self.dis[x * self.n + i] + w + self.dis[y * self.n + j],
                          self.dis[y * self.n + i] + w + self.dis[x * self.n + j])
                self.dis[x * self.n + y] = self.dis[y * self.n + x] = min(cur, self.dis[x * self.n + y])
        return

    def add_directed_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        for x in range(self.n):
            if self.dis[x * self.n + i] == self.inf:
                continue
            for y in range(self.n):
                cur = self.dis[x * self.n + i] + w + self.dis[j * self.n + y]
                self.dis[x * self.n + y] = min(cur, self.dis[x * self.n + y])
        return

    def get_cnt_of_shortest_path_undirected(self, mod=-1):
        self.cnt = [0] * self.n * self.n
        for i in range(self.n):
            self.cnt[i * self.n + i] = 1
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.dis[i * self.n + j] < self.inf:
                    self.cnt[i * self.n + j] = self.cnt[j * self.n + i] = 1
        for k in range(self.n):
            for i in range(self.n):
                if self.dis[i * self.n + k] == self.inf:
                    continue
                for j in range(i + 1, self.n):
                    if self.dis[i * self.n + k] + self.dis[k * self.n + j] < self.dis[j * self.n + i]:
                        self.dis[i * self.n + j] = self.dis[j * self.n + i] = self.dis[i * self.n + k] + self.dis[
                            k * self.n + j]
                        self.cnt[i * self.n + j] = self.cnt[j * self.n + i] = self.cnt[i * self.n + k] * self.cnt[
                            k * self.n + j]

                    elif self.dis[i * self.n + k] + self.dis[k * self.n + j] == self.dis[j * self.n + i]:
                        self.cnt[i * self.n + j] += self.cnt[i * self.n + k] * self.cnt[k * self.n + j]
                        self.cnt[j * self.n + i] += self.cnt[i * self.n + k] * self.cnt[k * self.n + j]
                        if mod != -1:
                            self.cnt[i * self.n + j] %= mod
                            self.cnt[j * self.n + i] %= mod
        return

    def get_cnt_of_shortest_path_directed(self, mod=-1):
        self.cnt = [0] * self.n * self.n
        for i in range(self.n):
            self.cnt[i * self.n + i] = 1
        for i in range(self.n):
            for j in range(self.n):
                if self.dis[i * self.n + j] < self.inf:
                    self.cnt[i * self.n + j] = self.cnt[j * self.n + i] = 1
        for k in range(self.n):
            for i in range(self.n):
                if self.dis[i * self.n + k] == self.inf:
                    continue
                for j in range(self.n):
                    if self.dis[i * self.n + k] + self.dis[k * self.n + j] < self.dis[j * self.n + i]:
                        self.dis[i * self.n + j] = self.dis[i * self.n + k] + self.dis[
                            k * self.n + j]
                        self.cnt[i * self.n + j] = self.cnt[i * self.n + k] * self.cnt[
                            k * self.n + j]

                    elif self.dis[i * self.n + k] + self.dis[k * self.n + j] == self.dis[j * self.n + i]:
                        self.cnt[i * self.n + j] += self.cnt[i * self.n + k] * self.cnt[k * self.n + j]
                        if mod != -1:
                            self.cnt[i * self.n + j] %= mod
        return

    def update_point_undirected(self, k):
        for i in range(self.n):
            if self.dis[i * self.n + k] == self.inf:
                continue
            for j in range(i + 1, self.n):
                cur = self.dis[i * self.n + k] + self.dis[k * self.n + j]
                self.dis[i * self.n + j] = self.dis[j * self.n + i] = min(self.dis[i * self.n + j], cur)
        return

    def update_point_directed(self, k):
        for i in range(self.n):
            if self.dis[i * self.n + k] == self.inf:
                continue
            for j in range(self.n):
                cur = self.dis[i * self.n + k] + self.dis[k * self.n + j]
                self.dis[i * self.n + j] = min(self.dis[i * self.n + j], cur)
        return

    def get_nodes_between_src_and_dst(self, i, j):
        path = [x for x in range(self.n) if
                self.dis[i * self.n + x] + self.dis[x * self.n + j] == self.dis[i * self.n + j]]
        return path
import math
from collections import deque
from heapq import heappop, heappush

from src.data_structure.tree_array.template import PointDescendPreMin
from src.graph.union_find.template import UnionFind



class ManhattanMST:
    def __init__(self):
        return

    @staticmethod
    def build(points):
        n = len(points)
        edges = list()

        def build():
            pos.sort()
            tree.initialize()
            mid = dict()
            for xx, yy, i in pos:
                val = tree.pre_min(dct[yy - xx] + 1)
                if val < math.inf:
                    edges.append((val + yy + xx, i, mid[val]))
                tree.point_descend(dct[yy - xx] + 1, -yy - xx)
                mid[-yy - xx] = i
            return

        nodes = set()
        for x, y in points:
            nodes.add(y - x)
            nodes.add(x - y)
            nodes.add(x + y)
            nodes.add(-y - x)
        nodes = sorted(nodes)
        dct = {num: i for i, num in enumerate(nodes)}
        m = len(dct)
        tree = PointDescendPreMin(m)
        pos = [(x, y, i) for i, (x, y) in enumerate(points)]
        build()
        pos = [(y, x, i) for i, (x, y) in enumerate(points)]
        build()
        pos = [(-y, x, i) for i, (x, y) in enumerate(points)]
        build()
        pos = [(x, -y, i) for i, (x, y) in enumerate(points)]
        build()

        uf = UnionFind(n)
        edges.sort()
        select = []
        ans = 0
        weight = []
        for w, u, v in edges:
            if uf.union(u, v):
                ans += w
                select.append((u, v))
                weight.append(w)
                if uf.part == 1:
                    break
        return ans, select, weight


class KruskalMinimumSpanningTree:
    def __init__(self, edges, n, method="kruskal"):
        self.n = n
        self.edges = edges
        self.cost = 0
        self.cnt = 0
        self.gen_minimum_spanning_tree(method)
        return

    def gen_minimum_spanning_tree(self, method):

        if method == "kruskal":
            # Edge priority
            self.edges.sort(key=lambda item: item[2])
            # greedy selection of edges based on weight for connected merging
            uf = UnionFind(self.n)
            for x, y, z in self.edges:
                if uf.union(x, y):
                    self.cost += z
            if uf.part != 1:
                self.cost = -1
        else:  # prim
            # Point priority with Dijkstra
            dct = [dict() for _ in range(self.n)]
            for i, j, w in self.edges:
                c = dct[i].get(j, math.inf)
                c = c if c < w else w
                dct[i][j] = dct[j][i] = c
            dis = [math.inf] * self.n
            dis[0] = 0
            visit = [0] * self.n
            stack = [(0, 0)]
            while stack:
                d, i = heappop(stack)
                if visit[i]:
                    continue
                visit[i] = 1
                # cost of mst
                self.cost += d
                # number of connected node
                self.cnt += 1
                for j in dct[i]:
                    w = dct[i][j]
                    if w < dis[j]:
                        dis[j] = w
                        heappush(stack, (w, j))
        return


class PrimMinimumSpanningTree:
    def __init__(self, function):
        self.dis = function
        return

    def build(self, nums):
        n = len(nums)
        ans = nex = 0
        rest = set(list(range(1, n)))
        visit = [math.inf] * n
        visit[nex] = 0
        while rest:
            i = nex
            rest.discard(i)
            d = visit[i]
            ans += d
            nex = -1
            x1, y1 = nums[i]
            for j in rest:
                x2, y2 = nums[j]
                dj = self.dis(x1, y1, x2, y2)
                if dj < visit[j]:
                    visit[j] = dj
                if nex == -1 or visit[j] < visit[nex]:
                    nex = j
        return ans if ans < math.inf else -1


class TreeAncestorMinIds:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.parent = [-1]
        self.order = 0
        self.start = [-1]
        self.end = [-1]
        self.parent = [-1]
        self.depth = [0]
        self.order_to_node = [-1]
        self.cols = max(2, math.ceil(math.log2(self.n)))
        self.min_ids = [self.n + 1] * self.n * self.cols * 10
        self.father = [-1] * self.n * self.cols
        self.ids = []
        return

    def add_directed_edge(self, u, v):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_from.append(u)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v)
        self.add_directed_edge(v, u)
        return

    def build_multiplication(self, ids):
        self.ids = ids
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        stack = [0]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    # the self.order of son nodes can be assigned for lexicographical self.order
                    if j != self.parent[i]:
                        self.parent[j] = i
                        self.depth[j] = self.depth[i] + 1
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex

        for i in range(self.n):
            self.father[i * self.cols] = self.parent[i]
            cur = ids[i * 10:i * 10 + 10]
            if self.parent[i] != -1:
                cur = self.update(cur, ids[self.parent[i] * 10:self.parent[i] * 10 + 10])
            self.min_ids[(i * self.cols) * 10:(i * self.cols) * 10 + 10] = cur[:]
        for j in range(1, self.cols):
            for i in range(self.n):
                father = self.father[i * self.cols + j - 1]
                if father != -1:
                    self.min_ids[(i * self.cols + j) * 10: (i * self.cols + j) * 10 + 10] = self.update(
                        self.min_ids[(i * self.cols + j - 1) * 10: (i * self.cols + j - 1) * 10 + 10],
                        self.min_ids[(father * self.cols + j - 1) * 10: (father * self.cols + j - 1) * 10 + 10])
                    self.father[i * self.cols + j] = self.father[father * self.cols + j - 1]
        return

    def update(self, lst1, lst2):
        lst = []

        m, n = len(lst1), len(lst2)
        i = j = 0
        while i < m and j < n and len(lst) < 10:
            if lst1[i] < lst2[j]:
                if not lst or lst[-1] < lst1[i]:
                    lst.append(lst1[i])
                i += 1
            else:
                if not lst or lst[-1] < lst2[j]:
                    lst.append(lst2[j])
                j += 1
        while i < m and len(lst) < 10:
            if not lst or lst[-1] < lst1[i]:
                lst.append(lst1[i])
            i += 1
        while j < n and len(lst) < 10:
            if not lst or lst[-1] < lst2[j]:
                lst.append(lst2[j])
            j += 1
        while len(lst) < 10:
            lst.append(self.n + 1)
        return lst[:10]

    def get_min_ids_between_nodes(self, x: int, y: int):
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        ans = self.update(self.ids[x * 10:x * 10 + 10], self.ids[y * 10:y * 10 + 10])
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = self.update(ans, self.min_ids[(x * self.cols + int(math.log2(d))) * 10:(x * self.cols + int(
                math.log2(d))) * 10 + 10])
            x = self.father[x * self.cols + int(math.log2(d))]

        if x == y:
            return ans

        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.father[x * self.cols + k] != self.father[y * self.cols + k]:
                ans = self.update(ans, self.min_ids[(x * self.cols + k) * 10:(x * self.cols + k) * 10 + 10])
                ans = self.update(ans, self.min_ids[(y * self.cols + k) * 10:(y * self.cols + k) * 10 + 10])
                x = self.father[x * self.cols + k]
                y = self.father[y * self.cols + k]

        ans = self.update(ans, self.min_ids[(x * self.cols) * 10:(x * self.cols) * 10 + 10])
        ans = self.update(ans, self.min_ids[(y * self.cols) * 10:(y * self.cols) * 10 + 10])
        return ans


class TreeMultiplicationMaxSecondWeights:
    def __init__(self, n, strictly=True):
        # strictly_second_minimum_spanning_tree
        self.n = n
        self.strictly = strictly
        self.point_head = [0] * (self.n + 1)
        self.edge_weight = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.depth = [0]
        self.cols = max(2, math.ceil(math.log2(self.n)))
        self.weights = [-1]
        self.father = [-1]
        return

    def add_directed_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_weight.append(w)
        self.edge_from.append(u)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v, w)
        self.add_directed_edge(v, u, w)
        return

    def build_multiplication(self):
        self.weights = [-1] * self.n * self.cols * 2
        self.father = [-1] * self.n * self.cols
        self.depth = [0] * self.n
        stack = [0]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != self.father[i * self.cols]:
                        self.father[j * self.cols] = i
                        self.weights[j * self.cols * 2: j * self.cols * 2 + 2] = [self.edge_weight[ind], -1]
                        self.depth[j] = self.depth[i] + 1
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex

        for j in range(1, self.cols):
            for i in range(self.n):
                father = self.father[i * self.cols + j - 1]
                if father != -1:
                    self.weights[(i * self.cols + j) * 2:(i * self.cols + j) * 2 + 2] = self.update(
                        self.weights[(i * self.cols + j - 1) * 2:(i * self.cols + j - 1) * 2 + 2],
                        self.weights[(father * self.cols + j - 1) * 2:(father * self.cols + j - 1) * 2 + 2])
                    self.father[i * self.cols + j] = self.father[father * self.cols + j - 1]
        return

    def update(self, lst1, lst2):
        a, b = lst1
        if not self.strictly:
            for x in lst2:
                if x >= a:
                    a, b = x, a
                elif x >= b:  # this is not strictly
                    b = x
        else:
            for x in lst2:
                if x > a:
                    a, b = x, a
                elif a > x > b:  # this is strictly
                    b = x
        return [a, b]

    def get_max_weights_between_nodes(self, x: int, y: int):
        assert 0 <= x < self.n
        assert 0 <= y < self.n
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        ans = [-1, -1]
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = self.update(ans, self.weights[
                                   (x * self.cols + int(math.log2(d))) * 2:(x * self.cols + int(math.log2(d))) * 2 + 2])
            x = self.father[x * self.cols + int(math.log2(d))]
        if x == y:
            return ans
        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.father[x * self.cols + k] != self.father[y * self.cols + k]:
                ans = self.update(ans, self.weights[(x * self.cols + k) * 2:(x * self.cols + k) * 2 + 2])
                ans = self.update(ans, self.weights[(y * self.cols + k) * 2:(y * self.cols + k) * 2 + 2])
                x = self.father[x * self.cols + k]
                y = self.father[y * self.cols + k]
        ans = self.update(ans, self.weights[(x * self.cols) * 2:(x * self.cols) * 2 + 2])
        ans = self.update(ans, self.weights[(y * self.cols) * 2:(y * self.cols) * 2 + 2])
        return ans


class TreeMultiplicationMaxWeights:
    def __init__(self, n):
        # second_mst|strictly_second_minimum_spanning_tree
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_weight = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.depth = [0]
        self.cols = max(2, math.ceil(math.log2(self.n)))
        self.weights = [-1]
        self.father = [-1]
        return

    def add_directed_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_weight.append(w)
        self.edge_from.append(u)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v, w):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v, w)
        self.add_directed_edge(v, u, w)
        return

    def build_multiplication(self):
        self.weights = [0] * self.n * self.cols
        self.father = [-1] * self.n * self.cols
        self.depth = [0] * self.n
        stack = [0]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != self.father[i * self.cols]:
                        self.father[j * self.cols] = i
                        self.weights[j * self.cols] = self.edge_weight[ind]
                        self.depth[j] = self.depth[i] + 1
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex

        for j in range(1, self.cols):
            for i in range(self.n):
                father = self.father[i * self.cols + j - 1]
                if father != -1:
                    self.weights[i * self.cols + j] = max(self.weights[i * self.cols + j - 1],
                                                          self.weights[father * self.cols + j - 1])
                    self.father[i * self.cols + j] = self.father[father * self.cols + j - 1]
        return

    def get_max_weights_between_nodes(self, x: int, y: int):
        assert 0 <= x < self.n
        assert 0 <= y < self.n
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        ans = 0
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = max(ans, self.weights[x * self.cols + int(math.log2(d))])
            x = self.father[x * self.cols + int(math.log2(d))]
        if x == y:
            return ans
        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.father[x * self.cols + k] != self.father[y * self.cols + k]:
                ans = max(ans, self.weights[x * self.cols + k])
                ans = max(ans, self.weights[y * self.cols + k])
                x = self.father[x * self.cols + k]
                y = self.father[y * self.cols + k]
        ans = max(ans, self.weights[x * self.cols])
        ans = max(ans, self.weights[y * self.cols])
        return ans

import heapq
from collections import deque




class DinicMaxflowMinCut:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_capacity = [0] * 2
        self.edge_to = [0] * 2
        self.edge_next = [0] * 2
        self.depth = [-1] * (self.n + 1)
        self.max_flow = 0
        self.min_cost = 0
        self.edge_id = 2
        self.cur = [0] * (self.n + 1)

    def _add_single_edge(self, u, v, cap):
        self.edge_capacity.append(cap)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_edge(self, u, v, cap):
        assert 1 <= u <= self.n
        assert 1 <= v <= self.n
        self._add_single_edge(u, v, cap)
        self._add_single_edge(v, u, 0)
        return

    def _bfs(self, s, t):
        for i in range(1, self.n + 1):
            self.depth[i] = -1
        self.depth[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            i = self.point_head[u]
            while i:
                v = self.edge_to[i]
                if self.edge_capacity[i] > 0 and self.depth[v] == -1:
                    self.depth[v] = self.depth[u] + 1
                    q.append(v)
                i = self.edge_next[i]
        return self.depth[t] != -1

    def _dfs(self, s, t, ff=math.inf):
        stack = [(s, ff, 0)]
        ind = 1
        max_flow = [0]
        sub = [-1]
        while stack:
            u, f, j = stack[-1]
            if u == t:
                max_flow[j] = f
                stack.pop()
                continue
            flag = 0
            while self.cur[u]:
                i = self.cur[u]
                v, cap, rev = self.edge_to[i], self.edge_capacity[i], i ^ 1
                if cap > 0 and self.depth[v] == self.depth[u] + 1:
                    x, y = f - max_flow[j], cap
                    x = x if x < y else y
                    if sub[j] == -1:
                        stack.append((v, x, ind))
                        max_flow.append(0)
                        sub[j] = ind
                        sub.append(-1)
                        ind += 1
                        flag = 1
                        break
                    else:
                        df = max_flow[sub[j]]
                        if df > 0:
                            self.edge_capacity[i] -= df
                            self.edge_capacity[i ^ 1] += df
                            max_flow[j] += df
                            if max_flow[j] == f:
                                break
                        sub[j] = -1

                self.cur[u] = self.edge_next[i]
            if flag:
                continue
            stack.pop()
        return max_flow[0]

    def max_flow_min_cut(self, s, t):
        total_flow = 0
        while self._bfs(s, t):
            for i in range(1, self.n + 1):
                self.cur[i] = self.point_head[i]
            flow = self._dfs(s, t)
            while flow > 0:
                total_flow += flow
                flow = self._dfs(s, t)
        return total_flow


class UndirectedGraph:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_w = [0] * 2
        self.edge_p = [0] * 2
        self.edge_to = [0] * 2
        self.edge_next = [0] * 2
        self.edge_id = 2

    def _add_single_edge(self, u, v, w, p):
        assert 1 <= u <= self.n
        assert 1 <= v <= self.n
        self.edge_w.append(w)
        self.edge_p.append(p)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_edge(self, u, v, w, p):
        assert 1 <= u <= self.n
        assert 1 <= v <= self.n
        self._add_single_edge(v, u, w, p)
        self._add_single_edge(u, v, w, p)
        return


class DirectedGraph:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_w = [0] * 2
        self.edge_to = [0] * 2
        self.edge_next = [0] * 2
        self.edge_id = 2

    def add_single_edge(self, u, v, w):
        assert 1 <= u <= self.n
        assert 1 <= v <= self.n
        self.edge_w.append(w)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def get_edge_ids(self, u):
        assert 1 <= u <= self.n
        i = self.point_head[u]
        ans = []
        while i:
            ans.append(i)
            i = self.edge_next[i]
        return


class DinicMaxflowMinCost:
    def __init__(self, n):
        self.n = n
        self.vis = [0] * (self.n + 1)
        self.point_head = [0] * (self.n + 1)
        self.edge_capacity = [0] * 2
        self.edge_cost = [0] * 2
        self.edge_to = [0] * 2
        self.edge_next = [0] * 2
        self.h = [math.inf] * (self.n + 1)
        self.dis = [math.inf] * (self.n + 1)
        self.max_flow = 0
        self.min_cost = 0
        self.edge_id = 2
        self.pre_edge = [0] * (self.n + 1)
        self.pre_point = [0] * (self.n + 1)

    def _add_single_edge(self, u, v, cap, c):
        self.edge_capacity.append(cap)
        self.edge_cost.append(c)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_edge(self, u, v, cap, c):
        assert 1 <= u <= self.n
        assert 1 <= v <= self.n
        self._add_single_edge(u, v, cap, c)
        self._add_single_edge(v, u, 0, -c)
        return

    def _spfa(self, s):
        self.h[s] = 0
        q = deque([s])
        self.vis[s] = 1
        while q:
            u = q.popleft()
            self.vis[u] = 0
            i = self.point_head[u]
            while i:
                v = self.edge_to[i]
                if self.edge_capacity[i] > 0 and self.h[v] > self.h[u] + self.edge_cost[i]:
                    self.h[v] = self.h[u] + self.edge_cost[i]
                    if not self.vis[v]:
                        q.append(v)
                        self.vis[v] = 1
                i = self.edge_next[i]
        return

    def _dijkstra(self, s, t):
        for i in range(1, self.n + 1):
            self.dis[i] = math.inf
            self.vis[i] = 0
        self.dis[s] = 0
        q = [(0, s)]
        while q:
            d, u = heapq.heappop(q)
            if self.vis[u]:
                continue
            self.vis[u] = 1
            i = self.point_head[u]
            while i:
                v = self.edge_to[i]
                nc = self.h[u] - self.h[v] + self.edge_cost[i]
                if self.edge_capacity[i] > 0 and self.dis[v] > self.dis[u] + nc:
                    self.dis[v] = self.dis[u] + nc
                    self.pre_edge[v] = i
                    self.pre_point[v] = u
                    if not self.vis[v]:
                        heapq.heappush(q, (self.dis[v], v))
                i = self.edge_next[i]
        return self.dis[t] < math.inf

    def max_flow_min_cost(self, s, t):
        self._spfa(s)
        while self._dijkstra(s, t):
            for i in range(1, self.n + 1):
                self.h[i] += self.dis[i]

            cur_flow = math.inf
            v = t
            while v != s:
                i = self.pre_edge[v]
                c = self.edge_capacity[i]
                cur_flow = cur_flow if cur_flow < c else c
                v = self.pre_point[v]

            v = t
            while v != s:
                i = self.pre_edge[v]
                self.edge_capacity[i] -= cur_flow
                self.edge_capacity[i ^ 1] += cur_flow
                v = self.pre_point[v]

            self.max_flow += cur_flow
            self.min_cost += cur_flow * self.h[t]

        return self.max_flow, self.min_cost
class PruferAndTree:
    def __init__(self):
        """默认以0为最小标号"""
        return

    @staticmethod
    def adj_to_parent(adj, root):

        def dfs(v):
            for u in adj[v]:
                if u != parent[v]:
                    parent[u] = v
                    dfs(u)

        n = len(adj)
        parent = [-1] * n
        dfs(root)
        return parent

    @staticmethod
    def parent_to_adj(parent):
        n = len(parent)
        adj = [[] for _ in range(n)]
        for i in range(n):
            if parent[i] != -1:  # 即 i!=root
                adj[i].append(parent[i])
                adj[parent[i]].append(i)
        return parent

    def tree_to_prufer(self, adj, root):
        # 以root为根的带标号树生成prufer序列，adj为邻接关系
        parent = self.adj_to_parent(adj, root)
        n = len(adj)
        # 统计度数，以较小的叶子节点序号开始
        ptr = -1
        degree = [0] * n
        for i in range(0, n):
            degree[i] = len(adj[i])
            if degree[i] == 1 and ptr == -1:
                ptr = i

        # 生成prufer序列
        code = [0] * (n - 2)
        leaf = ptr
        for i in range(0, n - 2):
            nex = parent[leaf]
            code[i] = nex
            degree[nex] -= 1
            if degree[nex] == 1 and nex < ptr:
                leaf = nex
            else:
                ptr = ptr + 1
                while degree[ptr] != 1:
                    ptr = ptr + 1
                leaf = ptr
        return code

    @staticmethod
    def prufer_to_tree(code, root):
        # prufer序列生成以root为根的带标号树
        n = len(code) + 2

        # 根据度确定初始叶节点
        degree = [1] * n
        for i in code:
            degree[i] += 1
        ptr = 0
        while degree[ptr] != 1:
            ptr += 1
        leaf = ptr

        # 逆向工程还原
        adj = [[] for _ in range(n)]
        for v in code:
            adj[v].append(leaf)
            adj[leaf].append(v)
            degree[v] -= 1
            if degree[v] == 1 and v < ptr and v != root:
                leaf = v
            else:
                ptr += 1
                while degree[ptr] != 1:
                    ptr += 1
                leaf = ptr

        # 最后还由就是生成prufer序列剩下的根和叶子节点
        adj[leaf].append(root)
        adj[root].append(leaf)
        for i in range(n):
            adj[i].sort()
        return adj
from collections import deque




class SPFA:
    def __init__(self):
        return

    @staticmethod
    def negative_circle_edge(dct, src=0, initial=0):
        """
        determine whether there is a negative loop and find the shortest path
        """
        # Finding the shortest path distance with negative weight and the number of path edges
        n = len(dct)
        dis = [math.inf] * n
        # flag of node in stack or not
        visit = [False] * n
        # the number of edges by the shortest path
        cnt = [0] * n
        queue = deque([src])
        dis[src] = initial
        visit[src] = True
        while queue:
            u = queue.popleft()
            visit[u] = False
            for v, w in dct[u]:  # Chain forward stars support self loops and double edges
                if dis[v] > dis[u] + w:
                    dis[v] = dis[u] + w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        # there is at least one negative loop starting from the starting point
                        return True, dis, cnt
                    # If the adjacent node is not already in the queue
                    # add it to the queue
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        # there is no negative loop starting from the starting point
        return False, dis, cnt

    @staticmethod
    def positive_circle_edge(dct, src=0, initial=0):
        """
        determine whether there is a positive loop and find the longest path
        """
        # Finding the longest path distance with negative weight and the number of path edges
        n = len(dct)
        dis = [-math.inf] * n
        # flag of node in stack or not
        visit = [False] * n
        # the number of edges by the shortest path
        cnt = [0] * n
        queue = deque([src])
        dis[src] = initial
        visit[src] = True
        while queue:
            u = queue.popleft()
            visit[u] = False
            for v, w in dct[u]:  # Chain forward stars support self loops and double edges
                if dis[v] < dis[u] + w:
                    dis[v] = dis[u] + w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        # there is at least one negative loop starting from the starting point
                        return True, dis, cnt
                    # If the adjacent node is not already in the queue
                    # add it to the queue
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        # there is no negative loop starting from the starting point
        return False, dis, cnt

    @staticmethod
    def negative_circle_mul(dct, src=0, initial=0):
        """Determine if there is a ring with a product greater than 1"""
        n = len(dct)
        dis = [math.inf for _ in range(n)]
        visit = [False] * n
        cnt = [0] * n
        queue = deque([src])
        dis[src] = initial
        visit[src] = True

        while queue:
            u = queue.popleft()
            visit[u] = False
            for v, w in dct[u]:
                if dis[v] > dis[u] * w:
                    dis[v] = dis[u] * w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        return True, dis, cnt
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        return False, dis, cnt

    @staticmethod
    def positive_circle_mul(dct, src=0, initial=1):
        """Determine if there is a ring with a product greater than 1"""
        n = len(dct)
        dis = [0 for _ in range(n)]
        visit = [False] * n
        cnt = [0] * n
        queue = deque([src])
        dis[src] = initial
        visit[src] = True

        while queue:
            u = queue.popleft()
            visit[u] = False
            for v, w in dct[u]:
                if dis[v] < dis[u] * w:
                    dis[v] = dis[u] * w
                    cnt[v] = cnt[u] + 1
                    if cnt[v] >= n:
                        return True, dis, cnt
                    if not visit[v]:
                        queue.append(v)
                        visit[v] = True
        return False, dis, cnt
import math


class DirectedGraphForTarjanScc:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * self.n
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.node_scc_id = [0]
        self.edge_id = 1
        self.scc_id = 0
        self.original_edge = set()
        return

    def initialize_graph(self):
        self.point_head = [0] * self.n
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.original_edge = set()
        return

    def add_directed_original_edge(self, i, j):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.original_edge.add(i * self.n + j)
        return

    def add_directed_edge(self, i, j):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.edge_from.append(i)
        self.edge_to.append(j)
        self.edge_next.append(self.point_head[i])
        self.point_head[i] = self.edge_id
        self.edge_id += 1
        return

    def build_scc(self):
        for val in self.original_edge:
            i, j = val // self.n, val % self.n
            if i != j:
                self.add_directed_edge(i, j)
        dfs_id = 0
        order = [self.n] * self.n
        low = [self.n] * self.n
        visit = [0] * self.n
        out = []
        in_stack = [0] * self.n
        self.node_scc_id = [-1] * self.n
        parent = [-1] * self.n
        point_head = self.point_head[:]
        for node in range(self.n):
            if not visit[node]:
                stack = [node]
                while stack:
                    cur = stack[-1]
                    ind = point_head[cur]
                    if not visit[cur]:
                        visit[cur] = 1
                        order[cur] = low[cur] = dfs_id
                        dfs_id += 1
                        out.append(cur)
                        in_stack[cur] = 1
                    if not ind:
                        stack.pop()
                        if order[cur] == low[cur]:
                            while out:
                                top = out.pop()
                                in_stack[top] = 0
                                self.node_scc_id[top] = self.scc_id
                                if top == cur:
                                    break
                            self.scc_id += 1

                        cur, nex = parent[cur], cur
                        if cur != -1:
                            low[cur] = min(low[cur], low[nex])
                    else:
                        nex = self.edge_to[ind]
                        point_head[cur] = self.edge_next[ind]
                        if not visit[nex]:
                            parent[nex] = cur
                            stack.append(nex)
                        elif in_stack[nex]:
                            low[cur] = min(low[cur], order[nex])
        # topological_order is [self.scc_id-1,self.scc_id-2,...,0] ?
        return

    def get_scc_edge_degree(self):
        scc_edge = set()
        for i in range(self.n):
            ind = self.point_head[i]
            while ind:
                j = self.edge_to[ind]
                a, b = self.node_scc_id[i], self.node_scc_id[j]
                if a != b:
                    scc_edge.add(a * self.scc_id + b)
                ind = self.edge_next[ind]
        scc_degree = [0] * self.scc_id
        for val in scc_edge:
            scc_degree[val % self.scc_id] += 1
        return scc_edge, scc_degree

    def get_scc_edge_degree_reverse(self):
        scc_edge = set()
        scc_cnt = [0] * self.scc_id
        for i in range(self.n):
            ind = self.point_head[i]
            while ind:
                j = self.edge_to[ind]
                a, b = self.node_scc_id[i], self.node_scc_id[j]
                if a != b:
                    scc_edge.add(b * self.scc_id + a)
                ind = self.edge_next[ind]
            scc_cnt[self.node_scc_id[i]] += 1
        scc_degree = [0] * self.scc_id
        for val in scc_edge:
            scc_degree[val % self.scc_id] += 1
        return scc_edge, scc_degree, scc_cnt

    def get_scc_dag_dp(self):
        scc_edge = set()
        scc_cnt = [0] * self.scc_id
        for i in range(self.n):
            ind = self.point_head[i]
            while ind:
                j = self.edge_to[ind]
                a, b = self.node_scc_id[i], self.node_scc_id[j]
                if a != b:
                    scc_edge.add(a * self.scc_id + b)
                ind = self.edge_next[ind]
            scc_cnt[self.node_scc_id[i]] += 1
        self.initialize_graph()
        for val in scc_edge:
            self.add_directed_edge(val // self.scc_id, val % self.scc_id)
        scc_degree = [0] * self.scc_id
        for val in scc_edge:
            scc_degree[val % self.scc_id] += 1
        stack = [i for i in range(self.scc_id) if not scc_degree[i]]
        cur_cnt = scc_cnt[:]
        ans = 0
        while stack:
            nex = []
            for i in stack:
                ans += cur_cnt[i] * cur_cnt[i]
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    scc_degree[j] -= 1
                    ans += scc_cnt[i] * cur_cnt[j]
                    scc_cnt[j] += scc_cnt[i]
                    if not scc_degree[j]:
                        nex.append(j)

                    ind = self.edge_next[ind]
            stack = nex
        return ans

    def get_scc_node_id(self):
        scc_node_id = [[] for _ in range(self.scc_id)]
        for i in range(self.n):
            scc_node_id[self.node_scc_id[i]].append(i)
        return scc_node_id

    def get_scc_cnt(self):
        scc_cnt = [0] * self.scc_id
        for i in range(self.n):
            scc_cnt[self.node_scc_id[i]] += 1
        return scc_cnt

    def build_new_graph_from_scc_id_to_original_node(self):
        self.initialize_graph()
        for i in range(self.n):
            self.add_directed_edge(self.node_scc_id[i], i)
        return

    def get_original_out_node(self, i):
        ind = self.point_head[i]
        lst = []
        while ind:
            lst.append(self.edge_to[ind])
            ind = self.edge_next[ind]
        return lst


class Tarjan:
    def __init__(self):
        return

    # @staticmethod
    # def get_scc(n: int, edge):
    #     assert all(i not in edge[i] for i in range(n))
    #     assert all(len(set(edge[i])) == len(edge[i]) for i in range(n))
    #     dfs_id = 0
    #     order, low = [math.inf] * n, [math.inf] * n
    #     visit = [0] * n
    #     out = []
    #     in_stack = [0] * n
    #     scc_id = 0
    #     # nodes list of every scc_id part
    #     scc_node_id = []
    #     # index if original node and value is scc_id part
    #     node_scc_id = [-1] * n
    #     parent = [-1] * n
    #     for node in range(n):
    #         if not visit[node]:
    #             stack = [[node, 0]]
    #             while stack:
    #                 cur, ind = stack[-1]
    #                 if not visit[cur]:
    #                     visit[cur] = 1
    #                     order[cur] = low[cur] = dfs_id
    #                     dfs_id += 1
    #                     out.append(cur)
    #                     in_stack[cur] = 1
    #                 if ind == len(edge[cur]):
    #                     stack.pop()
    #                     if order[cur] == low[cur]:
    #                         while out:
    #                             top = out.pop()
    #                             in_stack[top] = 0
    #                             while len(scc_node_id) < scc_id + 1:
    #                                 scc_node_id.append(set())
    #                             scc_node_id[scc_id].add(top)
    #                             node_scc_id[top] = scc_id
    #                             if top == cur:
    #                                 break
    #                         scc_id += 1
    #
    #                     cur, nex = parent[cur], cur
    #                     if cur != -1:
    #                         low[cur] = low[cur] if low[cur] < low[nex] else low[nex]
    #                 else:
    #                     nex = edge[cur][ind]
    #                     stack[-1][-1] += 1
    #                     if not visit[nex]:
    #                         parent[nex] = cur
    #                         stack.append([nex, 0])
    #                     elif in_stack[nex]:
    #                         low[cur] = low[cur] if low[cur] < order[nex] else order[nex]
    #
    #     # new graph after scc
    #     new_dct = [set() for _ in range(scc_id)]
    #     for i in range(n):
    #         for j in edge[i]:
    #             a, b = node_scc_id[i], node_scc_id[j]
    #             if a != b:
    #                 new_dct[a].add(b)
    #     new_degree = [0] * scc_id
    #     for i in range(scc_id):
    #         for j in new_dct[i]:
    #             new_degree[j] += 1
    #     assert len(scc_node_id) == scc_id
    #     return scc_id, scc_node_id, node_scc_id

    @staticmethod
    def get_pdcc(n: int, edge):

        dfs_id = 0
        order, low = [math.inf] * n, [math.inf] * n
        visit = [False] * n
        out = []
        parent = [-1] * n
        # number of group
        group_id = 0
        # nodes list of every group part
        group_node = []
        # index is original node and value is group_id set
        # cut node belong to two or more group
        node_group_id = [set() for _ in range(n)]
        child = [0] * n
        for node in range(n):
            if not visit[node]:
                stack = [[node, 0]]
                while stack:
                    cur, ind = stack[-1]
                    if not visit[cur]:
                        visit[cur] = True
                        order[cur] = low[cur] = dfs_id
                        dfs_id += 1

                    if ind == len(edge[cur]):
                        stack.pop()
                        cur, nex = parent[cur], cur
                        if cur != -1:
                            low[cur] = low[cur] if low[cur] < low[nex] else low[nex]
                            # cut node with rooted or not-rooted
                            if (parent == -1 and child[cur] > 1) or (parent != -1 and low[nex] >= order[cur]):
                                while out:
                                    top = out.pop()
                                    while len(group_node) < group_id + 1:
                                        group_node.append(set())
                                    group_node[group_id].add(top[0])
                                    group_node[group_id].add(top[1])
                                    node_group_id[top[0]].add(group_id)
                                    node_group_id[top[1]].add(group_id)
                                    if top == (cur, nex):
                                        break
                                group_id += 1
                            # We add all the edges encountered during deep search to the stack
                            # and when we find a cut point
                            # Pop up all the edges that this cutting point goes down to
                            # and the points connected by these edges are a pair of dots
                    else:
                        nex = edge[cur][ind]
                        stack[-1][-1] += 1
                        if nex == parent[cur]:
                            continue
                        if not visit[nex]:
                            parent[nex] = cur
                            out.append((cur, nex))
                            child[cur] += 1
                            stack.append([nex, 0])
                        elif low[cur] > order[nex]:
                            low[cur] = order[nex]
                            out.append((cur, nex))
            if out:
                while out:
                    top = out.pop()
                    group_node[group_id].add(top[0])
                    group_node[group_id].add(top[1])
                    node_group_id[top[0]].add(group_id)
                    node_group_id[top[1]].add(group_id)
                group_id += 1
        return group_id, group_node, node_group_id

    def get_edcc(self, n: int, edge):
        _, cutting_edges = self.get_cut(n, [list(e) for e in edge])
        for i, j in cutting_edges:
            edge[i].discard(j)
            edge[j].discard(i)
        # Remove all cut edges and leaving only edge doubly connected components
        # process the cut edges and then perform bfs on the entire undirected graph
        visit = [0] * n
        edcc_node_id = []
        for i in range(n):
            if visit[i]:
                continue
            stack = [i]
            visit[i] = 1
            cur = [i]
            while stack:
                x = stack.pop()
                for j in edge[x]:
                    if not visit[j]:
                        visit[j] = 1
                        stack.append(j)
                        cur.append(j)
            edcc_node_id.append(cur[:])

        # new graph after edcc
        edcc_id = len(edcc_node_id)
        node_edcc_id = [-1] * n
        for i, ls in enumerate(edcc_node_id):
            for x in ls:
                node_edcc_id[x] = i
        new_dct = [[] for _ in range(edcc_id)]
        for i in range(n):
            for j in edge[i]:
                a, b = node_edcc_id[i], node_edcc_id[j]
                if a != b:
                    new_dct[a].append(b)
        new_degree = [0] * edcc_id
        for i in range(edcc_id):
            for j in new_dct[i]:
                new_degree[j] += 1
        return edcc_node_id

    @staticmethod
    def get_cut(n: int, edge):
        order, low = [math.inf] * n, [math.inf] * n
        visit = [0] * n
        cutting_point = set()
        cutting_edge = []
        child = [0] * n
        parent = [-1] * n
        dfs_id = 0
        for i in range(n):
            if not visit[i]:
                stack = [[i, 0]]
                while stack:
                    cur, ind = stack[-1]
                    if not visit[cur]:
                        visit[cur] = 1
                        order[cur] = low[cur] = dfs_id
                        dfs_id += 1
                    if ind == len(edge[cur]):
                        stack.pop()
                        cur, nex = parent[cur], cur
                        if cur != -1:
                            pa = parent[cur]
                            low[cur] = low[cur] if low[cur] < low[nex] else low[nex]
                            if low[nex] > order[cur]:
                                cutting_edge.append((cur, nex) if cur < nex else (nex, cur))
                            if pa != -1 and low[nex] >= order[cur]:
                                cutting_point.add(cur)
                            elif pa == -1 and child[cur] > 1:
                                cutting_point.add(cur)
                    else:
                        nex = edge[cur][ind]
                        stack[-1][-1] += 1
                        if nex == parent[cur]:
                            continue
                        if not visit[nex]:
                            parent[nex] = cur
                            child[cur] += 1
                            stack.append([nex, 0])
                        else:
                            low[cur] = low[cur] if low[cur] < order[nex] else order[nex]
        return cutting_point, cutting_edge
import math



class GraphForTopologicalSort:
    def __init__(self, n, inf=math.inf):
        self.n = n
        self.inf = inf
        self.point_head = [0] * self.n
        self.degree = [0] * self.n
        self.edge_weight = [0]
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        return

    def add_directed_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.edge_weight.append(w)
        self.edge_from.append(i)
        self.edge_to.append(j)
        self.edge_next.append(self.point_head[i])
        self.point_head[i] = self.edge_id
        self.degree[j] += 1
        self.edge_id += 1
        return

    def add_undirected_edge(self, i, j, w):
        assert 0 <= i < self.n
        assert 0 <= j < self.n
        self.add_directed_edge(i, j, w)
        self.add_directed_edge(j, i, w)
        return

    def topological_sort_for_dag_dp(self, weights):
        ans = [0] * self.n
        stack = [i for i in range(self.n) if not self.degree[i]]
        for i in stack:
            ans[i] = weights[i]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    self.degree[j] -= 1
                    ans[j] = max(ans[j], ans[i] + weights[j])
                    if not self.degree[j]:
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex
        return ans

    def topological_sort_for_dag_dp_with_edge_weight(self, weights):
        ans = [0] * self.n
        stack = [i for i in range(self.n) if not self.degree[i]]
        for i in stack:
            ans[i] = weights[i]
        while stack:
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    self.degree[j] -= 1
                    ans[j] = max(ans[j], ans[i] + weights[j] + self.edge_weight[ind])
                    if not self.degree[j]:
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex
        return ans

    def topological_order(self):
        ans = []
        stack = [i for i in range(self.n) if not self.degree[i]]
        while stack:
            ans.extend(stack)
            nex = []
            for i in stack:
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    self.degree[j] -= 1
                    if not self.degree[j]:
                        nex.append(j)
                    ind = self.edge_next[ind]
            stack = nex
        return ans


class TopologicalSort:
    def __init__(self):
        return

    @staticmethod
    def get_rank(n, edges):
        dct = [list() for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            degree[j] += 1
            dct[i].append(j)
        stack = [i for i in range(n) if not degree[i]]
        visit = [-1] * n
        step = 0
        while stack:
            for i in stack:
                visit[i] = step
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
            step += 1
        return visit

    @staticmethod
    def count_dag_path(n, edges):
        # Calculate the number of paths in a directed acyclic connected graph
        edge = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            edge[i].append(j)
            degree[j] += 1
        cnt = [0] * n
        stack = [i for i in range(n) if not degree[i]]
        for x in stack:
            cnt[x] = 1
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    cnt[j] += cnt[i]
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return cnt

    @staticmethod
    def is_topology_unique(dct, degree, n):
        # Determine whether it is unique while ensuring the existence of topological sorting
        ans = []
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            ans.extend(stack)
            if len(stack) > 1:
                return False
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return True

    @staticmethod
    def is_topology_loop(edge, degree, n):
        # using Topological Sorting to Determine the Existence of Rings in a Directed Graph
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return all(x == 0 for x in degree)

    @staticmethod
    def bfs_topologic_order(n, dct, degree):
        # topological sorting determines whether there are rings in a directed graph
        # while recording the topological order of nodes
        order = [0] * n
        stack = [i for i in range(n) if degree[i] == 0]
        ind = 0
        while stack:
            nex = []
            for i in stack:
                order[i] = ind
                ind += 1
                for j in dct[i]:
                    degree[j] -= 1
                    if degree[j] == 0:
                        nex.append(j)
            stack = nex[:]
        if any(d > 0 for d in degree):
            return []
        return order


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
import math


class GraphDiameter:
    def __init__(self):
        return

    @staticmethod
    def get_diameter(dct, root=0):
        n = len(dct)
        dis = [math.inf] * n
        stack = [root]
        dis[root] = 0
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    if dis[j] == math.inf:
                        dis[j] = dis[i] + 1
                        nex.append(j)
            stack = nex[:]
        root = dis.index(max(dis))
        dis = [math.inf] * n
        stack = [root]
        dis[root] = 0
        while stack:
            nex = []
            for i in stack:
                for j in dct[i]:
                    if dis[j] == math.inf:
                        dis[j] = dis[i] + 1
                        nex.append(j)
            stack = nex[:]
        return max(dis)


class TreeDiameter:
    def __init__(self, dct):
        self.n = len(dct)
        self.dct = dct
        return

    def get_bfs_dis(self, root):
        dis = [math.inf] * self.n
        stack = [root]
        dis[root] = 0
        parent = [-1] * self.n
        while stack:
            i = stack.pop()
            for j, w in self.dct[i]:  # weighted edge
                if j != parent[i]:
                    parent[j] = i
                    dis[j] = dis[i] + w
                    stack.append(j)
        return dis, parent

    def get_diameter_info(self):
        """get tree diameter detail by weighted bfs twice"""
        dis, _ = self.get_bfs_dis(0)
        x = dis.index(max(dis))
        dis, parent = self.get_bfs_dis(x)
        y = dis.index(max(dis))
        path = [y]
        while path[-1] != x:
            path.append(parent[path[-1]])
        path.reverse()
        return x, y, path, dis[y]
class TreeDiffArray:

    def __init__(self):
        # node and edge differential method on tree
        return

    @staticmethod
    def bfs_iteration(dct, queries, root=0):
        """node differential method"""
        n = len(dct)
        stack = [root]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        diff = [0] * n
        for u, v, ancestor in queries:
            # update on the path u to ancestor and v to ancestor
            diff[u] += 1
            diff[v] += 1
            diff[ancestor] -= 1
            if parent[ancestor] != -1:
                diff[parent[ancestor]] -= 1

        # differential summation from bottom to top
        stack = [root]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    if j != parent[i]:
                        stack.append(j)
            else:
                i = ~i
                for j in dct[i]:
                    if j != parent[i]:
                        diff[i] += diff[j]
        return diff

    @staticmethod
    def bfs_iteration_edge(dct, queries, root=0):
        # Differential calculation of edges on the tree
        # where the count of edge is dropped to the corresponding down node
        n = len(dct)
        stack = [root]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        # Perform edge difference counting
        diff = [0] * n
        for u, v, ancestor in queries:
            # update the edge on the path u to ancestor and v to ancestor
            diff[u] += 1
            diff[v] += 1
            # make the down node represent the edge count
            diff[ancestor] -= 2

        # differential summation from bottom to top
        stack = [[root, 1]]
        while stack:
            i, state = stack.pop()
            if state:
                stack.append([i, 0])
                for j in dct[i]:
                    if j != parent[i]:
                        stack.append([j, 1])
            else:
                for j in dct[i]:
                    if j != parent[i]:
                        diff[i] += diff[j]
        return diff
import math
from collections import deque
from typing import List




class UnionFindGetLCA:
    def __init__(self, parent, root=0):
        n = len(parent)
        self.root_or_size = [-1] * n
        self.edge = [0] * n
        self.order = [0] * n
        assert parent[root] == root

        out_degree = [0] * n
        que = deque()
        for i in range(n):
            out_degree[parent[i]] += 1
        for i in range(n):
            if out_degree[i] == 0:
                que.append(i)

        for i in range(n - 1):
            v = que.popleft()
            fa = parent[v]
            x, y = self.union(v, fa)
            self.edge[y] = fa
            self.order[y] = i
            out_degree[fa] -= 1
            if out_degree[fa] == 0:
                que.append(fa)

        self.order[self.find(root)] = n
        return

    def union(self, v, fa):
        x, y = self.find(v), self.find(fa)
        if self.root_or_size[x] > self.root_or_size[y]:
            x, y = y, x
        self.root_or_size[x] += self.root_or_size[y]
        self.root_or_size[y] = x
        return x, y

    def find(self, v):
        while self.root_or_size[v] >= 0:
            v = self.root_or_size[v]
        return v

    def get_lca(self, u, v):
        lca = v
        while u != v:
            if self.order[u] < self.order[v]:
                u, v = v, u
            lca = self.edge[v]
            v = self.root_or_size[v]
        return lca


class UnionFindLCA:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.order = [0] * n
        return

    def find(self, x: int) -> int:
        lst = []
        while x != self.root[x]:
            lst.append(x)
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        # union to the smaller dfs order
        if self.order[root_x] < self.order[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        return True


class OfflineLCA:
    def __init__(self):
        return

    @staticmethod
    def bfs_iteration(dct, queries, root=0):
        """Offline query of LCA"""
        n = len(dct)
        ans = [dict() for _ in range(n)]
        for i, j in queries:  # Node pairs that need to be queried
            ans[i][j] = -1
            ans[j][i] = -1
        ind = 1
        stack = [root]
        # 0 is not visited
        # 1 is visited but not visit all its subtree nodes
        # 2 is visited included all its subtree nodes
        visit = [0] * n
        parent = [-1] * n
        uf = UnionFindLCA(n)
        depth = [0] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                uf.order[i] = ind  # dfs order
                ind += 1
                visit[i] = 1
                stack.append(~i)
                for j in dct[i]:
                    if j != parent[i]:
                        parent[j] = i
                        depth[j] = depth[i] + 1
                        stack.append(j)
                for y in ans[i]:
                    if visit[y] == 1:
                        ans[y][i] = ans[i][y] = y
                    else:
                        ans[y][i] = ans[i][y] = uf.find(y)
            else:
                i = ~i
                visit[i] = 2
                uf.union(i, parent[i])

        return [ans[i][j] for i, j in queries]


class TreeAncestorPool:

    def __init__(self, edges, weight):
        # node 0 as root
        n = len(edges)
        self.n = n
        self.parent = [-1] * n
        self.depth = [-1] * n
        stack = deque([0])
        self.depth[0] = 0
        while stack:
            i = stack.popleft()
            for j in edges[i]:
                if self.depth[j] == -1:
                    self.depth[j] = self.depth[i] + 1
                    self.parent[j] = i
                    stack.append(j)

        # Set the number of layers based on the node size
        self.cols = max(2, math.ceil(math.log2(n)))
        self.dp = [[-1] * self.cols for _ in range(n)]
        self.weight = [[math.inf] * self.cols for _ in range(n)]
        for i in range(n):
            # the amount of water accumulated during weight maintenance
            self.dp[i][0] = self.parent[i]
            self.weight[i][0] = weight[i]

        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i][j - 1]
                if father != -1:
                    self.dp[i][j] = self.dp[father][j - 1]
                    self.weight[i][j] = self.weight[father][j - 1] + self.weight[i][j - 1]
        return

    def get_final_ancestor(self, node: int, v: int) -> int:
        # query the final math.inflow of water into the pool with Multiplication method
        for i in range(self.cols - 1, -1, -1):
            if v > self.weight[node][i]:
                v -= self.weight[node][i]
                node = self.dp[node][i]
        return node


class TreeAncestor:

    def __init__(self, edges, root=0):
        n = len(edges)
        self.parent = [-1] * n
        self.depth = [-1] * n
        stack = [root]
        self.depth[root] = 0
        while stack:
            i = stack.pop()
            for j in edges[i]:
                if self.depth[j] == -1:
                    self.depth[j] = self.depth[i] + 1  # can change to be weighted
                    self.parent[j] = i
                    stack.append(j)

        self.cols = max(2, math.ceil(math.log2(n)))
        self.dp = [-1] * self.cols * n
        for i in range(n):
            self.dp[i * self.cols] = self.parent[i]

        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i * self.cols + j - 1]
                if father != -1:
                    self.dp[i * self.cols + j] = self.dp[father * self.cols + j - 1]
        return

    def get_kth_ancestor(self, node: int, k: int) -> int:
        for i in range(self.cols - 1, -1, -1):
            if k & (1 << i):
                node = self.dp[node * self.cols + i]
                if node == -1:
                    break
        return node

    def get_lca(self, x: int, y: int) -> int:
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            x = self.dp[x * self.cols + int(math.log2(d))]
        if x == y:
            return x
        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x * self.cols + k] != self.dp[y * self.cols + k]:
                x = self.dp[x * self.cols + k]
                y = self.dp[y * self.cols + k]
        return self.dp[x * self.cols]

    def get_dist(self, u: int, v: int) -> int:
        lca = self.get_lca(u, v)
        depth_u = self.depth[u]
        depth_v = self.depth[v]
        depth_lca = self.depth[lca]
        return depth_u + depth_v - 2 * depth_lca


class TreeCentroid:
    def __init__(self):
        return

    @staticmethod
    def centroid_finder(to, root=0):
        # recursive centroid partitioning of rooted trees
        centroids = []
        pre_cent = []
        subtree_size = []
        n = len(to)
        roots = [(root, -1, 1)]
        size = [1] * n
        is_removed = [0] * n
        parent = [-1] * n
        while roots:
            root, pc, update = roots.pop()
            parent[root] = -1
            if update:
                stack = [root]
                dfs_order = []
                while stack:
                    u = stack.pop()
                    size[u] = 1
                    dfs_order.append(u)
                    for v in to[u]:
                        if v == parent[u] or is_removed[v]:
                            continue
                        parent[v] = u
                        stack.append(v)
                for u in dfs_order[::-1]:
                    if u == root:
                        break
                    size[parent[u]] += size[u]
            c = root
            while 1:
                mx, u = size[root] // 2, -1
                for v in to[c]:
                    if v == parent[c] or is_removed[v]:
                        continue
                    if size[v] > mx:
                        mx, u = size[v], v
                if u == -1:
                    break
                c = u
            centroids.append(c)
            pre_cent.append(pc)
            subtree_size.append(size[root])
            is_removed[c] = 1
            for v in to[c]:
                if is_removed[v]:
                    continue
                roots.append((v, c, v == parent[c]))
        # the centroid array of the tree
        # the parent node corresponding to the centroid
        # the size of the subtree with the centroid as the root
        return centroids, pre_cent, subtree_size



class TreeAncestorMaxSubNode:
    def __init__(self, x):
        self.val = self.pref = self.suf = max(x, 0)
        self.sm = x
        pass



class TreeAncestorMaxSub:
    def __init__(self, edges: List[List[int]], values):
        n = len(edges)
        self.values = values
        self.parent = [-1] * n
        self.depth = [-1] * n
        stack = deque([0])
        self.depth[0] = 0
        while stack:
            i = stack.popleft()
            for j in edges[i]:
                if self.depth[j] == -1:
                    self.depth[j] = self.depth[i] + 1
                    self.parent[j] = i
                    stack.append(j)

        self.cols = max(2, math.ceil(math.log2(n)))
        self.dp = [-1] * self.cols * n
        self.weight = [TreeAncestorMaxSubNode(0)] * self.cols * n
        for i in range(n):
            self.dp[i * self.cols] = self.parent[i]
            if i:
                self.weight[i * self.cols] = TreeAncestorMaxSubNode(values[self.parent[i]])
            else:
                self.weight[i * self.cols] = TreeAncestorMaxSubNode(0)

        for j in range(1, self.cols):
            for i in range(n):
                father = self.dp[i * self.cols + j - 1]
                pre = self.weight[i * self.cols + j - 1]
                if father != -1:
                    self.dp[i * self.cols + j] = self.dp[father * self.cols + j - 1]
                    self.weight[i * self.cols + j] = self.merge(pre, self.weight[father * self.cols + j - 1])

        return

    @staticmethod
    def merge(a, b):
        ret = TreeAncestorMaxSubNode(0)
        ret.pref = max(a.pref, a.sm + b.pref)
        ret.suf = max(b.suf, b.sm + a.suf)
        ret.sm = a.sm + b.sm
        ret.val = max(a.val, b.val)
        ret.val = max(ret.val, a.suf + b.pref)
        return ret

    @staticmethod
    def reverse(a):
        a.suf, a.pref = a.pref, a.suf
        return a

    def get_ancestor_node_max(self, x: int, y: int) -> TreeAncestorMaxSubNode:
        ans = TreeAncestorMaxSubNode(self.values[x])
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            ans = self.merge(ans, self.weight[x * self.cols + int(math.log2(d))])
            x = self.dp[x * self.cols + int(math.log2(d))]
        return ans

    def get_kth_ancestor(self, node: int, k: int) -> int:
        for i in range(self.cols - 1, -1, -1):
            if k & (1 << i):
                node = self.dp[node * self.cols + i]
                if node == -1:
                    break
        return node

    def get_lca(self, x: int, y: int) -> int:
        if self.depth[x] < self.depth[y]:
            x, y = y, x
        while self.depth[x] > self.depth[y]:
            d = self.depth[x] - self.depth[y]
            x = self.dp[x * self.cols + int(math.log2(d))]
        if x == y:
            return x
        for k in range(int(math.log2(self.depth[x])), -1, -1):
            if self.dp[x * self.cols + k] != self.dp[y * self.cols + k]:
                x = self.dp[x * self.cols + k]
                y = self.dp[y * self.cols + k]
        return self.dp[x * self.cols]

    def get_max_con_sum(self, x: int, y: int) -> int:
        if x == y:
            return TreeAncestorMaxSubNode(self.values[x]).val
        z = self.get_lca(x, y)
        if z == x:
            ans = self.get_ancestor_node_max(y, x)
            return ans.val
        if z == y:
            ans = self.get_ancestor_node_max(x, y)
            return ans.val

        ax = self.get_kth_ancestor(x, self.depth[x] - self.depth[z])
        by = self.get_kth_ancestor(y, self.depth[y] - self.depth[z] - 1)
        a = self.get_ancestor_node_max(x, ax)
        b = self.get_ancestor_node_max(y, by)
        ans = self.merge(a, self.reverse(b))
        return ans.val

class HeavyChain:
    def __init__(self, dct, root=0) -> None:
        self.n = len(dct)
        self.dct = dct
        self.parent = [-1] * self.n  # father of node
        self.cnt_son = [1] * self.n  # number of subtree nodes
        self.weight_son = [-1] * self.n  # heavy son
        self.top = [-1] * self.n  # chain forward star
        self.dfn = [0] * self.n  # index is original node and value is its dfs order
        self.rev_dfn = [0] * self.n  # index is dfs order and value is its original node
        self.depth = [0] * self.n  # depth of node
        self.build_weight(root)
        self.build_dfs(root)
        return

    def build_weight(self, root) -> None:
        # get the math.info of heavy children of the tree
        stack = [root]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in self.dct[i]:
                    if j != self.parent[i]:
                        stack.append(j)
                        self.parent[j] = i
                        self.depth[j] = self.depth[i] + 1
            else:
                i = ~i
                for j in self.dct[i]:
                    if j != self.parent[i]:
                        self.cnt_son[i] += self.cnt_son[j]
                        if self.weight_son[i] == -1 or self.cnt_son[j] > self.cnt_son[self.weight_son[i]]:
                            self.weight_son[i] = j
        return

    def build_dfs(self, root) -> None:
        # get the math.info of dfs order
        stack = [(root, root)]
        order = 0
        while stack:
            i, tp = stack.pop()
            self.dfn[i] = order
            self.rev_dfn[order] = i
            self.top[i] = tp
            order += 1
            # visit heavy children first, then visit light children
            w = self.weight_son[i]
            for j in self.dct[i]:
                if j != self.parent[i] and j != w:
                    stack.append((j, j))
            if w != -1:
                stack.append((w, tp))
        return

    def query_chain(self, x, y):
        # query the shortest path from x to y that passes through the chain segment
        pre = []
        post = []
        while self.top[x] != self.top[y]:
            if self.depth[self.top[x]] > self.depth[self.top[y]]:
                pre.append([self.dfn[x], self.dfn[self.top[x]]])
                x = self.parent[self.top[x]]
            else:
                post.append([self.dfn[self.top[y]], self.dfn[y]])
                y = self.parent[self.top[y]]

        a, b = self.dfn[x], self.dfn[y]
        pre.append([a, b])
        lca = a if a < b else b
        pre += post[::-1]
        return pre, lca

    def query_lca(self, x, y):
        # query the LCA nearest common ancestor of x and y
        while self.top[x] != self.top[y]:
            if self.depth[self.top[x]] < self.depth[self.top[y]]:
                x, y = y, x
            x = self.parent[self.top[x]]
        # returned value is the actual number of the node and not the dfn!!!
        return x if self.depth[x] < self.depth[y] else yfrom collections import defaultdict




class UnionFind:
    def __init__(self, n: int) -> None:
        self.root_or_size = [-1] * n
        self.part = n
        self.n = n
        return

    def initialize(self):
        for i in range(self.n):
            self.root_or_size[i] = -1
        self.part = self.n
        return

    def find(self, x):
        y = x
        while self.root_or_size[x] >= 0:
            # range_merge_to_disjoint to the direct root node after query
            x = self.root_or_size[x]
        while y != x:
            self.root_or_size[y], y = x, self.root_or_size[y]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.root_or_size[root_x] < self.root_or_size[root_y]:
            root_x, root_y = root_y, root_x
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
        self.part -= 1
        return True

    def union_left(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        self.root_or_size[root_x] += self.root_or_size[root_y]
        self.root_or_size[root_y] = root_x
        self.part -= 1
        return True

    def union_right(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
        self.part -= 1
        return True

    def union_max(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if root_x > root_y:
            root_x, root_y = root_y, root_x
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
        self.part -= 1
        return

    def union_min(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if root_x < root_y:
            root_x, root_y = root_y, root_x
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
        self.part -= 1
        return

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def size(self, x):
        return -self.root_or_size[self.find(x)]

    def get_root_part(self):
        # get the nodes list of every root
        part = defaultdict(list)
        n = len(self.root_or_size)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        # get the size of every root
        size = defaultdict(int)
        n = len(self.root_or_size)
        for i in range(n):
            if self.find(i) == i:
                size[i] = -self.root_or_size[i]
        return size


class UnionFindGeneral:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n
        return

    def find(self, x):
        y = x
        while x != self.root[x]:
            # range_merge_to_disjoint to the direct root node after query
            x = self.root[x]
        while y != x:
            self.root[y], y = x, self.root[y]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] >= self.size[root_y]:  # merge to bigger size part
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        self.size[root_x] = 0
        self.part -= 1
        return True

    def union_right(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if root_x > root_y:  # merge to bigger root number
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        self.size[root_x] = 0
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        part = defaultdict(list)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        size = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
        return size


class UnionFindWeighted:
    def __init__(self, n):
        self.root_or_size = [-1] * n
        self.dis = [0] * n

    def find(self, x):  # distance to the root
        stack = []
        while self.root_or_size[x] >= 0:
            stack.append(x)
            x = self.root_or_size[x]
        d = 0
        while stack:
            y = stack.pop()
            self.root_or_size[y] = x
            self.dis[y] += d
            d = self.dis[y]
        return x

    def union(self, x, y, val):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.root_or_size[root_x] > self.root_or_size[root_y]:
                self.root_or_size[root_y] += self.root_or_size[root_x]
                self.root_or_size[root_x] = root_y
                self.dis[root_x] = val + self.dis[y] - self.dis[x]  # distance to the root
            else:
                self.root_or_size[root_x] += self.root_or_size[root_y]
                self.root_or_size[root_y] = root_x
                self.dis[root_y] = -val + self.dis[x] - self.dis[y]  # distance to the root
        elif self.dis[x] - self.dis[y] != val:
            return True
        return False

    def union_right_weight(self, x, y, val):
        root_x = self.find(x)
        root_y = self.find(y)
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
        self.dis[root_x] = val + self.dis[y] - self.dis[x]  # distance to the root
        return True

    def union_right(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        size = self.root_or_size[root_y]
        self.root_or_size[root_y] += self.root_or_size[root_x]
        self.root_or_size[root_x] = root_y
        self.dis[root_x] += size
        return True

    def size(self, x):
        return -self.root_or_size[self.find(x)]

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)


class PersistentUnionFind:
    def __init__(self, n):
        self.rank = [0] * n
        self.root = list(range(n))
        self.version = [math.inf] * n

    def union(self, x, y, tm):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.version[root_y] = tm
                self.root[root_y] = root_x
            else:
                self.version[root_x] = tm
                self.root[root_x] = root_y
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1
            return True
        return False

    def find(self, x, tm=math.inf):
        while not (x == self.root[x] or self.version[x] >= tm):
            x = self.root[x]
        return x

    def is_connected(self, x, y, tm):
        return self.find(x, tm) == self.find(y, tm)


class UnionFindSP:
    def __init__(self, n: int) -> None:
        self.root = list(range(n))
        self.size = [0] * n
        self.height = [0] * n
        self.n = n
        return

    def find(self, x):
        y = x
        while x != self.root[x]:
            # range_merge_to_disjoint to the direct root node after query
            x = self.root[x]
        while y != x:
            self.root[y], y = x, self.root[y]
        return x

    def union_right(self, x, y, h):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return 0
        if root_x > root_y:
            root_x, root_y = root_y, root_x
        self.size[root_y] += self.size[root_x]
        self.height[root_y] += self.height[root_x]
        self.root[root_x] = root_y
        if root_y == self.n - 1:
            return h * self.size[root_x] - self.height[root_x]
        return 0


class UnionFindInd:
    def __init__(self, n: int, cnt) -> None:
        self.root_or_size = [-1] * n * cnt
        self.part = [n] * cnt
        self.n = n
        return

    def initialize(self, ind):
        for i in range(self.n):
            self.root_or_size[ind * self.n + i] = -1
        self.part[ind] = self.n
        return

    def find(self, x, ind):
        y = x
        while self.root_or_size[ind * self.n + x] >= 0:
            # range_merge_to_disjoint to the direct root node after query
            x = self.root_or_size[ind * self.n + x]
        while y != x:
            self.root_or_size[ind * self.n + y], y = x, self.root_or_size[ind * self.n + y]
        return x

    def union(self, x, y, ind):
        root_x = self.find(x, ind)
        root_y = self.find(y, ind)
        if root_x == root_y:
            return False
        if self.root_or_size[ind * self.n + root_x] < self.root_or_size[ind * self.n + root_y]:
            root_x, root_y = root_y, root_x
        self.root_or_size[ind * self.n + root_y] += self.root_or_size[ind * self.n + root_x]
        self.root_or_size[ind * self.n + root_x] = root_y
        self.part[ind] -= 1
        return True

    def is_connected(self, x, y, ind):
        return self.find(x, ind) == self.find(y, ind)
import math


class BrainStorming:
    def __init__(self):
        return

    @staticmethod
    def minimal_coin_need(n, m, nums):
        # there are n selectable and math.infinite coins
        # and the minimum number of coins required to form all combinations of 1-m
        nums += [m + 1]
        nums.sort()
        if nums[0] != 1:
            return -1
        ans = sum_ = 0
        for i in range(n):
            nex = nums[i + 1] - 1
            nex = nex if nex < m else m
            x = math.ceil((nex - sum_) / nums[i])
            x = x if x >= 0 else 0
            ans += x
            sum_ += x * nums[i]
            if sum_ >= m:
                break
        return ans
import bisect
from bisect import bisect_left
from collections import deque, defaultdict



class LongestIncreasingSubsequence:
    def __init__(self):
        return

    @staticmethod
    def definitely_increase(nums):
        # longest strictly increasing subsequence
        dp = []
        for num in nums:
            i = bisect.bisect_left(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    @staticmethod
    def definitely_not_reduce(nums):
        # longest non-decreasing subsequence
        dp = []
        for num in nums:
            i = bisect.bisect_right(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    def definitely_reduce(self, nums):
        # longest strictly decreasing subsequence
        nums = [-num for num in nums]
        return self.definitely_increase(nums)

    def definitely_not_increase(self, nums):
        # longest strictly non-increasing subsequence
        nums = [-num for num in nums]
        return self.definitely_not_reduce(nums)


class LcsComputeByLis:
    def __init__(self):
        return

    def length_of_lcs(self, s1, s2) -> int:
        """compute lcs with lis"""
        # O(n**2)
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m - 1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])

        return self.length_of_lis(nums)

    @staticmethod
    def length_of_lis(nums) -> int:
        # greedy and binary search to check lis
        stack = []
        for x in nums:
            idx = bisect_left(stack, x)
            if idx < len(stack):
                stack[idx] = x
            else:
                stack.append(x)
        # length of lis
        return len(stack)

    def index_of_lcs(self, s1, s2):
        # greedy and binary search to check lis output the index
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m - 1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])
        # return the index of lcs in s2
        res = self.minimum_lexicographical_order_of_lis(nums)
        return res

    @staticmethod
    def minimum_lexicographical_order_of_lis(nums):
        """template of minimum lexicographical order lis"""
        # greedy and binary search to check lis output the index
        if not nums:
            return []
        n = len(nums)
        tops = [nums[0]]
        piles = [0] * n
        piles[0] = 0

        for i in range(1, n):
            if nums[i] > tops[-1]:
                piles[i] = len(tops)
                tops.append(nums[i])
            else:
                j = bisect.bisect_left(tops, nums[i])
                piles[i] = j
                tops[j] = nums[i]

        lis = []
        j = len(tops) - 1
        for i in range(n - 1, -1, -1):
            if piles[i] == j:
                lis.append(nums[i])
                j -= 1
        lis.reverse()
        return lis

    @staticmethod
    def length_and_max_sum_of_lis(nums):
        """the maximum sum of lis with maximum length"""
        # which can be extended to non-decreasing non-increasing minimum sum and so on
        dp = []
        q = []
        for num in nums:
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                q[length].append([num, num])
            else:
                while q[length - 1] and q[length - 1][0][0] >= num:
                    q[length - 1].popleft()
                cur = q[length - 1][0][1] + num
                while q[length] and q[length][-1][1] <= cur:
                    q[length].pop()
                q[length].append([num, cur])
        return q[-1][0][1]

    @staticmethod
    def length_and_cnt_of_lis(nums):
        """template to finding the number of LIS"""
        # O(nlogn)
        dp = []  # LIS array
        s = []  # index is length and value is sum
        q = []  # index is length and value is [num, cnt]
        for num in nums:
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(s) <= len(dp):
                s.append(0)
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                s[length] += 1
                q[length].append([num, 1])
            else:
                while q[length - 1] and q[length - 1][0][0] >= num:
                    s[length - 1] -= q[length - 1].popleft()[1]
                s[length] += s[length - 1]
                q[length].append([num, s[length - 1]])
        return s[-1]

    @staticmethod
    def length_and_cnt_of_lcs(s1, s2, mod=10 ** 9 + 7):
        """template of number of lcs calculated by lis"""
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = dict()
        for i in range(m - 1, -1, -1):
            if s2[i] not in mapper:
                mapper[s2[i]] = []
            mapper[s2[i]].append(i)

        dp = []
        s = []
        q = []
        for c in s1:
            for num in mapper.get(c, []):
                if not dp or num > dp[-1]:
                    dp.append(num)
                    length = len(dp)
                else:
                    i = bisect.bisect_left(dp, num)
                    dp[i] = num
                    length = i + 1
                while len(s) <= len(dp):
                    s.append(0)
                while len(q) <= len(dp):
                    q.append(deque())

                if length == 1:
                    s[length] += 1
                    q[length].append((num, 1))
                else:
                    while q[length - 1] and q[length - 1][0][0] >= num:
                        s[length - 1] -= q[length - 1].popleft()[1]
                    s[length] += s[length - 1]
                    s[length] %= mod
                    q[length].append((num, s[length - 1]))
        return len(dp), s[-1]from typing import List

from src.data_structure.sorted_list.template import SortedList


class MinimumPairXor:
    def __init__(self):
        """
        if x < y < z then min(x^y, y^z) < x^z, thus the minimum xor pair must be adjacent
        """
        self.lst = SortedList()
        self.xor = SortedList()
        return

    def add(self, num):
        i = self.lst.bisect_left(num)
        if i < len(self.lst):
            if 0 <= i - 1:
                self.xor.discard(self.lst[i] ^ self.lst[i - 1])
        self.lst.add(num)
        if 0 <= i - 1 < len(self.lst):
            self.xor.add(num ^ self.lst[i - 1])
        if 0 <= i + 1 < len(self.lst):
            self.xor.add(num ^ self.lst[i + 1])
        return

    def remove(self, num):
        i = self.lst.bisect_left(num)
        if 0 <= i - 1 < len(self.lst):
            self.xor.discard(num ^ self.lst[i - 1])
        if 0 <= i + 1 < len(self.lst):
            self.xor.discard(num ^ self.lst[i + 1])
        self.lst.discard(num)
        if i < len(self.lst) and i - 1 >= 0:
            self.xor.add(self.lst[i] ^ self.lst[i - 1])
        return

    def query(self):
        return self.xor[0]


class BitOperation:
    def __init__(self):
        return

    @staticmethod
    def sum_xor(n):
        """xor num of range(0, x+1)"""
        if n % 4 == 0:
            return n  # (4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0
        elif n % 4 == 1:
            return 1  # n^(n-1)
        elif n % 4 == 2:
            return n + 1  # n^(n-1)^(n-2)
        return 0  # n^(n-1)^(n-2)^(n-3)

    @staticmethod
    def graycode_to_integer(graycode):
        graycode_len = len(graycode)
        binary = list()
        binary.append(graycode[0])
        for i in range(1, graycode_len):
            if graycode[i] == binary[i - 1]:
                b = 0
            else:
                b = 1
            binary.append(str(b))
        return int("0b" + ''.join(binary), 2)

    @staticmethod
    def integer_to_graycode(integer):
        binary = bin(integer).replace('0b', '')
        graycode = list()
        binary_len = len(binary)
        graycode.append(binary[0])
        for i in range(1, binary_len):
            if binary[i - 1] == binary[i]:
                g = 0
            else:
                g = 1
            graycode.append(str(g))
        return ''.join(graycode)

    @staticmethod
    def get_graycode(n: int) -> List[int]:
        """all graycode number whose length small or equal to n"""
        code = [0, 1]
        for i in range(1, n):
            code.extend([(1 << i) + num for num in code[::-1]])
        return code
from src.data_structure.sorted_list.template import SortedList
from src.data_structure.tree_array.template import PointAddRangeSum


class CantorExpands:
    def __init__(self, n, mod=0):
        self.mod = mod
        self.perm = [1] * (n + 1)
        for i in range(2, n):
            if mod:
                self.perm[i] = i * self.perm[i - 1] % mod
            else:
                self.perm[i] = i * self.perm[i - 1]
        return

    def array_to_rank(self, nums):
        """"permutation rank of nums"""
        n = len(nums)
        out = 1
        lst = SortedList(nums)
        for i in range(n):
            fact = self.perm[n - i - 1]
            res = lst.bisect_left(nums[i])
            lst.discard(nums[i])
            out += res * fact
            if self.mod:
                out %= self.mod
        return out

    def array_to_rank_with_tree(self, nums):
        """"permutation rank of nums"""
        n = len(nums)
        out = 1
        tree = PointAddRangeSum(n)
        tree.build([1] * n)
        for i in range(n):
            fact = self.perm[n - i - 1]
            res = tree.range_sum(0, nums[i] - 2) if nums[i] >= 2 else 0
            tree.point_add(nums[i] - 1, -1)
            out += res * fact
            if self.mod:
                out %= self.mod
        return out


    def rank_to_array(self, n, k):
        """"nums with permutation rank k"""
        nums = list(range(1, n + 1))
        ans = []
        while k and nums:
            single = self.perm[len(nums) - 1]
            i = (k - 1) // single
            ans.append(nums.pop(i))
            k -= i * single
        return ans
class Combinatorics:
    def __init__(self, n, mod):
        assert mod > n
        self.n = n + 10
        self.mod = mod

        self.perm = [1]
        self.rev = [1]
        self.inv = [0]
        self.fault = [0]

        self.build_perm()
        self.build_rev()
        self.build_inv()
        self.build_fault()
        return

    def build_perm(self):
        self.perm = [1] * (self.n + 1)  # (i!) % mod
        for i in range(1, self.n + 1):
            self.perm[i] = self.perm[i - 1] * i % self.mod
        return

    def build_rev(self):
        self.rev = [1] * (self.n + 1)  # pow(i!, -1, mod)
        self.rev[-1] = pow(self.perm[-1], -1, self.mod)  # GcdLike().mod_reverse(self.perm[-1], self.mod)
        for i in range(self.n - 1, 0, -1):
            self.rev[i] = (self.rev[i + 1] * (i + 1) % self.mod)  # pow(i!, -1, mod)
        return

    def build_inv(self):
        self.inv = [0] * (self.n + 1)  # pow(i, -1, mod)
        self.inv[1] = 1
        for i in range(2, self.n + 1):
            self.inv[i] = (self.mod - self.mod // i) * self.inv[self.mod % i] % self.mod
        return

    def build_fault(self):
        self.fault = [0] * (self.n + 1)  # fault permutation
        self.fault[0] = 1
        self.fault[2] = 1
        for i in range(3, self.n + 1):
            self.fault[i] = (i - 1) * (self.fault[i - 1] + self.fault[i - 2])
            self.fault[i] %= self.mod
        return

    def comb(self, a, b):
        if a < b:
            return 0
        res = self.perm[a] * self.rev[b] * self.rev[a - b]  # comb(a, b) % mod = (a!/(b!(a-b)!)) % mod
        return res % self.mod

    def factorial(self, a):
        res = self.perm[a]  # (a!) % mod
        return res % self.mod

    def inverse(self, n):
        res = self.perm[n - 1] * self.rev[n] % self.mod  # pow(n, -1, mod)
        return res

    def catalan(self, n):
        res = (self.comb(2 * n, n) - self.comb(2 * n, n - 1)) % self.mod
        return res


class Lucas:
    def __init__(self):
        # comb(a,b) % p
        return

    @staticmethod
    def comb(n, m, p):
        # comb(n, m) % p
        ans = 1
        for x in range(n - m + 1, n + 1):
            ans *= x
            ans %= p
        for x in range(1, m + 1):
            ans *= pow(x, -1, p)
            ans %= p
        return ans

    def lucas_iter(self, n, m, p):
        # math.comb(n, m) % p where p is prime
        if m == 0:
            return 1
        stack = [[n, m]]
        dct = dict()
        while stack:
            n, m = stack.pop()
            if n >= 0:
                if m == 0:
                    dct[(n, m)] = 1
                    continue
                stack.append((~n, m))
                stack.append((n // p, m // p))
            else:
                n = ~n
                dct[(n, m)] = (self.comb(n % p, m % p, p) % p) * dct[(n // p, m // p)] % p
        return dct[(n, m)]

    @staticmethod
    def extend_lucas(self, n, m, p):
        # math.comb(n, m) % p where p is not necessary prime
        return
import math
import random
from typing import List


class MinCircleOverlap:
    def __init__(self):
        self.pi = math.acos(-1)
        self.esp = 10 ** (-10)
        return

    def get_min_circle_overlap(self, points: List[List[int]]):
        # 随机增量法求解最小圆覆盖

        def cross(a, b):
            return a[0] * b[1] - b[0] * a[1]

        def intersection_point(p1, v1, p2, v2):
            # 求解两条直线的交点
            u = (p1[0] - p2[0], p1[1] - p2[1])
            t = cross(v2, u) / cross(v1, v2)
            return p1[0] + v1[0] * t, p1[1] + v1[1] * t

        def is_point_in_circle(circle_x, circle_y, circle_r, x, y):
            res = math.sqrt((x - circle_x) ** 2 + (y - circle_y) ** 2)
            if abs(res - circle_r) < self.esp:
                return True
            if res < circle_r:
                return True
            return False

        def vec_rotate(v, theta):
            x, y = v
            return x * math.cos(theta) + y * math.sin(theta), -x * math.sin(theta) + y * math.cos(theta)

        def get_out_circle(x1, y1, x2, y2, x3, y3):
            xx1, yy1 = (x1 + x2) / 2, (y1 + y2) / 2
            vv1 = vec_rotate((x2 - x1, y2 - y1), self.pi / 2)
            xx2, yy2 = (x1 + x3) / 2, (y1 + y3) / 2
            vv2 = vec_rotate((x3 - x1, y3 - y1), self.pi / 2)
            pp = intersection_point((xx1, yy1), vv1, (xx2, yy2), vv2)
            res = math.sqrt((pp[0] - x1) ** 2 + (pp[1] - y1) ** 2)
            return pp[0], pp[1], res

        random.shuffle(points)
        n = len(points)
        p = points

        # 圆心与半径
        cc1 = (p[0][0], p[0][1], 0)
        for ii in range(1, n):
            if not is_point_in_circle(cc1[0], cc1[1], cc1[2], p[ii][0], p[ii][1]):
                cc2 = (p[ii][0], p[ii][1], 0)
                for jj in range(ii):
                    if not is_point_in_circle(cc2[0], cc2[1], cc2[2], p[jj][0], p[jj][1]):
                        dis = math.sqrt((p[jj][0] - p[ii][0]) ** 2 + (p[jj][1] - p[ii][1]) ** 2)
                        cc3 = ((p[jj][0] + p[ii][0]) / 2, (p[jj][1] + p[ii][1]) / 2, dis / 2)
                        for kk in range(jj):
                            if not is_point_in_circle(cc3[0], cc3[1], cc3[2], p[kk][0], p[kk][1]):
                                cc3 = get_out_circle(p[ii][0], p[ii][1], p[jj][0], p[jj][1], p[kk][0], p[kk][1])
                        cc2 = cc3
                cc1 = cc2

        return cc1
from functools import reduce


class CRT:
    def __init__(self):
        return

    @staticmethod
    def chinese_remainder(pairs):
        """中国剩余定理"""
        mod_list, remainder_list = [p[0] for p in pairs], [p[1] for p in pairs]
        mod_product = reduce(lambda x, y: x * y, mod_list)
        mi_list = [mod_product // x for x in mod_list]
        mi_inverse = [ExtendCRT().exgcd(mi_list[i], mod_list[i])[0] for i in range(len(mi_list))]
        x = 0
        for i in range(len(remainder_list)):
            x += mi_list[i] * mi_inverse[i] * remainder_list[i]
            x %= mod_product
        return x


class ExtendCRT:
    # 在模数不coprime的情况下，最小的非负整数解
    def __init__(self):
        return

    def gcd(self, a, b):
        if b == 0:
            return a
        return self.gcd(b, a % b)

    def lcm(self, a, b):
        return a * b // self.gcd(a, b)

    def exgcd(self, a, b):
        if b == 0:
            return 1, 0
        x, y = self.exgcd(b, a % b)
        return y, x - a // b * y

    def uni(self, p, q):
        r1, m1 = p
        r2, m2 = q

        d = self.gcd(m1, m2)
        assert (r2 - r1) % d == 0
        # 否则无解
        l1, l2 = self.exgcd(m1 // d, m2 // d)

        return (r1 + (r2 - r1) // d * l1 * m1) % self.lcm(m1, m2), self.lcm(m1, m2)

    def pipline(self, eq):
        return reduce(self.uni, eq)



class FastPower:
    def __init__(self):
        return

    @staticmethod
    def fast_power_api(a, b, mod):
        return pow(a, b, mod)

    @staticmethod
    def fast_power(a, b, mod):
        a = a % mod
        res = 1
        while b > 0:
            if b & 1:
                res = res * a % mod
            a = a * a % mod
            b >>= 1
        return res

    @staticmethod
    def float_fast_pow(x: float, m: int) -> float:

        if m >= 0:
            res = 1
            while m > 0:
                if m & 1:
                    res *= x
                x *= x
                m >>= 1
            return res
        m = -m
        res = 1
        while m > 0:
            if m & 1:
                res *= x
            x *= x
            m >>= 1
        return 1.0 / res


class MatrixFastPowerFlatten:
    def __init__(self):
        return

    @staticmethod
    def matrix_pow_flatten(base, n, p, mod=10 ** 9 + 7):
        assert len(base) == n * n
        res = [0] * n * n
        ans = [0] * n * n
        for i in range(n):
            ans[i * n + i] = 1
        while p:
            if p & 1:
                for i in range(n):
                    for j in range(n):
                        cur = 0
                        for k in range(n):
                            cur += ans[i * n + k] * base[k * n + j]
                            cur %= mod
                        res[i * n + j] = cur
                for i in range(n):
                    for j in range(n):
                        ans[i * n + j] = res[i * n + j]
            for i in range(n):
                for j in range(n):
                    cur = 0
                    for k in range(n):
                        cur += base[i * n + k] * base[k * n + j]
                        cur %= mod
                    res[i * n + j] = cur
            for i in range(n):
                for j in range(n):
                    base[i * n + j] = res[i * n + j]
            p >>= 1
        return ans

class MatrixFastPowerMin:
    def __init__(self):
        return

    @staticmethod
    def _matrix_mul(a, b):
        n = len(a)
        res = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                res[i][j] = min(max(a[i][k], b[k][j]) for k in range(n))
        return res

    def matrix_pow(self, base, p):
        n = len(base)
        ans = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            ans[i][i] = 0
        while p:
            if p & 1:
                ans = self._matrix_mul(ans, base)
            base = self._matrix_mul(base, base)
            p >>= 1
        return ans


class MatrixFastPower:
    def __init__(self):
        return

    @staticmethod
    def _matrix_mul(a, b, mod=10 ** 9 + 7):
        n = len(a)
        res = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                res[i][j] = sum(a[i][k] * b[k][j] for k in range(n)) % mod
        return res

    def matrix_pow(self, base, p, mod=10 ** 9 + 7):
        n = len(base)
        ans = [[0] * n for _ in range(n)]
        for i in range(n):
            ans[i][i] = 1
        while p:
            if p & 1:
                ans = self._matrix_mul(ans, base, mod)
            base = self._matrix_mul(base, base, mod)
            p >>= 1
        return ans
class GcdLike:

    def __init__(self):
        return

    @staticmethod
    def extend_gcd(a, b):
        sub = dict()
        stack = [(a, b, 0)]
        while stack:
            a, b, s = stack.pop()
            if a == 0:
                sub[(a, b)] = (b, 0, 1) if b >= 0 else (-b, 0, -1)
                continue
            if s == 0:
                stack.append((a, b, 1))
                stack.append((b % a, a, 0))
            else:
                gcd, x, y = sub[(b % a, a)]
                sub[(a, b)] = (gcd, y - (b // a) * x, x) if gcd >= 0 else (-gcd, -y + (b // a) * x, -x)
                assert gcd == a * (y - (b // a) * x) + b * x
        return sub[(a, b)]

    @staticmethod
    def binary_gcd(a, b):
        if a == 0:
            return abs(b)
        if b == 0:
            return abs(a)
        a, b = abs(a), abs(b)
        c = 1
        while a - b:
            if a & 1:
                if b & 1:
                    if a > b:
                        a = (a - b) >> 1
                    else:
                        b = (b - a) >> 1
                else:
                    b = b >> 1
            else:
                if b & 1:
                    a = a >> 1
                else:
                    c = c << 1
                    b = b >> 1
                    a = a >> 1
        return c * a

    @staticmethod
    def general_gcd(x, y):
        while y:
            x, y = y, x % y
        return abs(x)

    def mod_reverse(self, a, p):
        g, x, y = self.extend_gcd(a, p)
        assert g == 1  # necessary of pow(a, -1, p)
        return (x + p) % p

    def solve_equation(self, a, b, n=1):
        """a * x + b * y = n"""
        gcd, x, y = self.extend_gcd(a, b)
        assert a * x + b * y == gcd
        if n % gcd:
            return []
        x0 = x * (n // gcd)
        y0 = y * (n // gcd)
        # xt = x0 + b // gcd * t (t=0,1,2,3,...)
        # yt = y0 - a // gcd * t (t=0,1,2,3,...)
        return [gcd, x0, y0]

    @staticmethod
    def add_to_n(n):

        # minimum times to make a == n or b == n by change [a, b] to [a + b, b] or [a, a + b] from [1, 1]
        if n == 1:
            return 0

        def gcd_minus(a, b, c):
            nonlocal ans
            if c >= ans or not b:
                return
            if b == 1:
                ans = ans if ans < c + a - 1 else c + a - 1
                return
            # reverse_thinking
            gcd_minus(b, a % b, c + a // b)
            return

        ans = n - 1
        for i in range(1, n):
            gcd_minus(n, i, 0)
        return ans
import math
import random
from typing import List

from src.data_structure.sorted_list.template import SortedList



class Geometry:
    def __init__(self):
        return

    @staticmethod
    def is_convex_quad(points):
        n = len(points)

        def cross_product():
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

        prev_cross_product = None
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]
            cp = cross_product()
            if prev_cross_product is None:
                prev_cross_product = cp
            elif cp * prev_cross_product < 0:
                return False
        return True

    @staticmethod
    def compute_center(x1, y1, x2, y2, r):
        # Calculate the centers of two circles passing through two different points and determining the radius
        px, py = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x1 - x2, y1 - y2
        h = math.sqrt(r * r - (dx * dx + dy * dy) / 4)
        res = []
        for fx, fy in ((1, -1), (-1, 1)):
            cx = px + fx * h * dy / math.sqrt(dx * dx + dy * dy)
            cy = py + fy * h * dx / math.sqrt(dx * dx + dy * dy)
            res.append([cx, cy])
        return res

    @staticmethod
    def same_line(point1, point2, point3):
        # calculating three point collinearity
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        return (x2 - x1) * (y3 - y2) == (x3 - x2) * (y2 - y1)

    @staticmethod
    def vertical_angle(p1, p2, p3):
        x1, y1 = p1[0] - p2[0], p1[1] - p2[1]
        x2, y2 = p3[0] - p2[0], p3[1] - p2[1]
        return x1 * x2 + y1 * y2 == 0

    @staticmethod
    def is_rectangle_overlap(rec1: List[int], rec2: List[int]) -> bool:
        x1, y1, x2, y2 = rec1  # left_down
        x3, y3, x4, y4 = rec2  # right_up
        if x2 <= x3 or x4 <= x1 or y4 <= y1 or y2 <= y3:
            return False
        return True

    @staticmethod
    def is_rectangle_separate(rec1: List[int], rec2: List[int]) -> bool:
        x1, y1, x2, y2 = rec1  # left_down
        a1, b1, a2, b2 = rec2  # right_up
        return x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1

    @staticmethod
    def is_rectangle_touching(rec1: List[int], rec2: List[int]) -> bool:
        x1, y1, x2, y2 = rec1  # left_down
        a1, b1, a2, b2 = rec2  # right_up
        edge_touch = ((x1 == a2 or x2 == a1) and (y1 < b2 and y2 > b1)) or \
                     ((y1 == b2 or y2 == b1) and (x1 < a2 and x2 > a1))

        # Check if one rectangle's corner is on the other rectangle
        corner_touch = (x1 == a2 and (y1 == b2 or y2 == b1)) or \
                       (x2 == a1 and (y1 == b2 or y2 == b1)) or \
                       (y1 == b2 and (x1 == a2 or x2 == a1)) or \
                       (y2 == b1 and (x1 == a2 or x2 == a1))

        return edge_touch or corner_touch

    @staticmethod
    def compute_slope(x1, y1, x2, y2):
        assert [x1, y1] != [x2, y2]
        # Determine the slope of a straight line based on two different points
        if x1 == x2:
            ans = (x1, 0)
        else:
            a = y2 - y1
            b = x2 - x1
            g = math.gcd(a, b)
            if b < 0:
                a *= -1
                b *= -1
            ans = (a // g, b // g)
        return ans

    @staticmethod
    def compute_square_point_non_vertical(x0, y0, x2, y2):
        # Given two points on the diagonal of a rectangle and ensuring that they are different
        # calculate the coordinates of the other two points
        x1 = (x0 + x2 + y2 - y0) / 2
        y1 = (y0 + y2 + x0 - x2) / 2
        x3 = (x0 + x2 - y2 + y0) / 2
        y3 = (y0 + y2 - x0 + x2) / 2  # not need to be vertical
        return (x1, y1), (x3, y3)

    @staticmethod
    def compute_square_point(x0, y0, x2, y2):
        # Given two points on the diagonal of a rectangle and ensuring that they are different
        # calculate the coordinates of the other two points
        assert [x0, y0] != [x2, y2]  # need to be vertical
        assert abs(x0 - x2) == abs(y0 - y2)
        return (x0, y2), (x2, y0)

    @staticmethod
    def compute_square_area(x0, y0, x2, y2):
        # Given the points on the diagonal of a square
        # calculate the area of the square, taking into account that it is an integer
        ans = (x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)
        return ans // 2

    @staticmethod
    def compute_triangle_area(x1, y1, x2, y2, x3, y3):
        # Can be used to determine the positional relationship between points and triangles
        return abs((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3)) / 2

    @staticmethod
    def compute_triangle_area_double(x1, y1, x2, y2, x3, y3):
        # Can be used to determine the positional relationship between points and triangles
        return abs((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3))

    @staticmethod
    def line_intersection_line(start1, end1, start2, end2):
        # Calculate the intersection point of two line segments that are bottommost and leftmost
        # If there is no intersection point, return empty
        x1, y1 = start1
        x2, y2 = end1
        x3, y3 = start2
        x4, y4 = end2
        det = lambda a, b, c, dd: a * dd - b * c
        d = det(x1 - x2, x4 - x3, y1 - y2, y4 - y3)
        p = det(x4 - x2, x4 - x3, y4 - y2, y4 - y3)
        q = det(x1 - x2, x4 - x2, y1 - y2, y4 - y2)
        if d != 0:
            lam, eta = p / d, q / d
            if not (0 <= lam <= 1 and 0 <= eta <= 1):
                return []
            return [lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2]
        if p != 0 or q != 0:
            return []
        t1, t2 = sorted([start1, end1]), sorted([start2, end2])
        if t1[1] < t2[0] or t2[1] < t1[0]:
            return []
        return max(t1[0], t2[0])

    @staticmethod
    def angle_with_x_axis(x, y):
        if x == 0:
            if y > 0:
                return 0.5 * math.pi
            if y < 0:
                return 1.5 * math.pi
        if y == 0:
            if x > 0:
                return 0
            if x < 0:
                return math.pi

        d = (x ** 2 + y ** 2) ** 0.5
        if y > 0:
            return math.acos(x * 1.0 / d)
        else:
            return 2 * math.pi - math.acos(x * 1.0 / d)

    @staticmethod
    def angle_between_vector(va, vb):
        x1, y1 = va
        x2, y2 = vb
        d1 = x1 ** 2 + y1 ** 2
        d2 = x2 ** 2 + y2 ** 2
        d3 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        if d1 + d2 - d3 > 0:
            return [(d1 + d2 - d3) ** 2, d1 * d2]
        else:
            return [-(d1 + d2 - d3) ** 2, d1 * d2]

    @staticmethod
    def circumscribed_circle_of_triangle(x1, y1, x2, y2, x3, y3):
        x = ((y2 - y1) * (y3 * y3 - y1 * y1 + x3 * x3 - x1 * x1) - (y3 - y1) * (
                y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1)) * 1.0 / (
                    2 * (x3 - x1) * (y2 - y1) - 2 * (x2 - x1) * (y3 - y1))

        y = ((x2 - x1) * (x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1) - (x3 - x1) * (
                x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1)) * 1.0 / (
                    2 * (y3 - y1) * (x2 - x1) - 2 * (y2 - y1) * (x3 - x1))

        r = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return x, y, r

    @staticmethod
    def get_circle_sector_angle(x1, y1, x2, y2, r):
        d_square = ((x1 - x2) ** 2 + (y1 - y2) ** 2)
        angle = math.acos(1 - d_square * 0.5 / r / r)
        return angle


class ClosetPair:
    def __init__(self):
        return

    @staticmethod
    def bucket_grid(n: int, nums):
        # Use random increment method to divide the grid and calculate the closest point pairs on the plane
        def dis(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        def check(p):
            nonlocal ss
            return p[0] // ss, p[1] // ss

        def update(buck, ind):
            nonlocal dct
            if buck not in dct:
                dct[buck] = []
            dct[buck].append(ind)
            return

        assert n >= 2
        random.shuffle(nums)

        # initialization
        dct = dict()
        ans = dis(nums[0], nums[1])
        ss = ans ** 0.5
        update(check(nums[0]), 0)
        update(check(nums[1]), 1)
        if ans == 0:
            return 0

        # traverse with random increments
        for i in range(2, n):
            a, b = check(nums[i])
            res = ans
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    cur = (x + a, y + b)
                    if cur in dct:
                        for j in dct[cur]:
                            now = dis(nums[i], nums[j])
                            res = res if res < now else now
            if res == 0:  # Directly return at a distance of 0
                return 0
            if res < ans:
                # initialization again
                ans = res
                ss = ans ** 0.5
                dct = dict()
                for x in range(i + 1):
                    update(check(nums[x]), x)
            else:
                update(check(nums[i]), i)
        # The return value is the square of the Euclidean distance
        return ans

    @staticmethod
    def divide_and_conquer(lst):

        # Using Divide and Conquer to Solve the Pairs of Nearest Points in a Plane
        lst.sort(key=lambda p: p[0])

        def distance(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        def conquer(point_set, min_dis, mid):
            smallest = min_dis
            point_mid = point_set[mid]
            pt = []
            for point in point_set:
                if (point[0] - point_mid[0]) ** 2 <= min_dis:
                    pt.append(point)
            pt.sort(key=lambda x: x[1])
            for i in range(len(pt)):
                for j in range(i + 1, len(pt)):
                    if (pt[i][1] - pt[j][1]) ** 2 >= min_dis:
                        break
                    cur = distance(pt[i], pt[j])
                    smallest = smallest if smallest < cur else cur
            return smallest

        def check(point_set):
            min_dis = math.inf
            if len(point_set) <= 3:
                n = len(point_set)
                for i in range(n):
                    for j in range(i + 1, n):
                        cur = distance(point_set[i], point_set[j])
                        min_dis = min_dis if min_dis < cur else cur
                return min_dis
            mid = len(point_set) // 2
            left = point_set[:mid]
            right = point_set[mid + 1:]
            min_left_dis = check(left)
            min_right_dis = check(right)
            min_dis = min_left_dis if min_left_dis < min_right_dis else min_right_dis
            range_merge_to_disjoint_dis = conquer(point_set, min_dis, mid)
            return min_dis if min_dis < range_merge_to_disjoint_dis else range_merge_to_disjoint_dis

        return check(lst)

    @staticmethod
    def sorted_pair(points) -> float:
        # Using an ordered list for calculating the closest point pairs on a plane
        def dis(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        points.sort(key=lambda p: [p[0], [1]])
        lst1 = SortedList()
        lst2 = SortedList()
        ans = math.inf
        ss = ans ** 0.5
        n = len(points)
        for i in range(n):
            x, y = points[i]
            while lst1 and abs(x - lst1[0][0]) >= ss:
                a, b = lst1.pop(0)
                lst2.discard((b, a))
            while lst1 and abs(x - lst1[-1][0]) >= ss:
                a, b = lst1.pop()
                lst2.discard((b, a))

            ind = lst2.bisect_left((y - ss, -math.inf))
            while ind < len(lst2) and abs(y - lst2[ind][0]) <= ss:
                res = dis([x, y], lst2[ind][::-1])
                ans = ans if ans < res else res
                ind += 1

            ss = ans ** 0.5
            lst1.add((x, y))
            lst2.add((y, x))
        return ans

    @staticmethod
    def bucket_grid_between_two_sets(n: int, nums1, nums2):

        # Using the Random Incremental Method to Divide Grids and
        # Calculate the Nearest Point Pairs of Two Planar Point Sets

        def dis(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        def check(p):
            nonlocal ss
            return p[0] // ss, p[1] // ss

        def update(buck, ind):
            nonlocal dct
            if buck not in dct:
                dct[buck] = []
            dct[buck].append(ind)
            return

        random.shuffle(nums1)
        random.shuffle(nums2)

        dct = dict()
        ans = math.inf
        for i in range(n):
            cur = dis(nums1[i], nums2[0])
            ans = cur if ans > cur else ans
        ss = ans ** 0.5
        if ans == 0:
            return 0
        for i in range(n):
            update(check(nums1[i]), i)

        for i in range(1, n):
            a, b = check(nums2[i])
            res = ans
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    cur = (x + a, y + b)
                    if cur in dct:
                        for j in dct[cur]:
                            now = dis(nums2[i], nums1[j])
                            res = res if res < now else now
            if res == 0:
                return 0
            if res < ans:
                ans = res
                ss = ans ** 0.5
                dct = dict()
                for x in range(n):
                    update(check(nums1[x]), x)
        # The return value is the square of the Euclidean distance
        return ans
import math
from decimal import Decimal
from typing import List


class HighPrecision:
    def __init__(self):
        return

    @staticmethod
    def factorial_factorial_suffix_zero_cnt(n):
        """Compute number of suffixes 0 with 1!*2!***n!"""
        ans = 0
        num = 5
        while num <= n:
            ans += num * (n // num) * (n // num - 1) // 2
            ans += (n // num) * (n % num + 1)
            num *= 5
        return ans

    @staticmethod
    def factorial_suffix_zero_cnt(n):
        """Compute number of suffixes 0 with n!"""
        ans = 0
        while n:
            ans += n // 5
            n //= 5
        return ans

    @staticmethod
    def float_pow(r, n):
        """High precision calculation of the power of decimals"""
        ans = (Decimal(r) ** int(n)).normalize()
        ans = "{:f}".format(ans)
        return ans

    @staticmethod
    def fraction_to_decimal(numerator: int, denominator: int) -> str:
        """Convert fractions to rational numbers or math.infinitely recurring decimals"""
        if numerator % denominator == 0:
            return str(numerator // denominator) + ".0"
        ans = []
        if numerator * denominator < 0:
            ans.append("-")
        numerator = abs(numerator)
        denominator = abs(denominator)

        ans.append(str(numerator // denominator))
        numerator %= denominator
        ans.append(".")
        reminder = numerator % denominator
        dct = dict()
        while reminder and reminder not in dct:
            dct[reminder] = len(ans)
            reminder *= 10
            ans.append(str(reminder // denominator))
            reminder %= denominator
        if reminder in dct:
            ans.insert(dct[reminder], "(")
            ans.append(")")
        return "".join(ans)

    @staticmethod
    def decimal_to_fraction(st):
        # Decimal to Fraction
        def sum_fraction(tmp):
            mu = tmp[0][1]
            for ls in tmp[1:]:
                mu = math.lcm(mu, ls[1])
            zi = sum(ls[0] * mu // ls[1] for ls in tmp)
            mz = math.gcd(mu, zi)
            return [zi // mz, mu // mz]

        # Converting rational numbers or math.infinite recurring decimals to fractions
        if "." in st:
            lst = st.split(".")
            integer = [int(lst[0]), 1] if lst[0] else [0, 1]
            if "(" not in lst[1]:
                non_repeat = [int(lst[1]), 10 ** len(lst[1])
                              ] if lst[1] else [0, 1]
                repeat = [0, 1]
            else:
                pre, post = lst[1].split("(")
                non_repeat = [int(pre), 10 ** len(pre)] if pre else [0, 1]
                post = post[:-1]
                repeat = [int(post), int("9" * len(post)) * 10 ** len(pre)]
        else:

            integer = [int(st), 1]
            non_repeat = [0, 1]
            repeat = [0, 1]
        return sum_fraction([integer, non_repeat, repeat])


class FloatToFrac:
    def __init__(self):
        # Convert floating-point operations to fractional calculations
        return

    @staticmethod
    def frac_add_without_gcd(frac1: List[int], frac2: List[int]) -> List[int]:
        # Both denominators a1 and a2 are required to be non-zero
        b1, a1 = frac1
        b2, a2 = frac2
        assert a1 != 0 and a2 != 0
        b = b1 * a2 + b2 * a1
        a = a1 * a2
        if a < 0:
            a *= -1
            b *= -1
        return [b, a]

    @staticmethod
    def frac_add(frac1: List[int], frac2: List[int]) -> List[int]:
        # Both denominators a1 and a2 are required to be non-zero
        b1, a1 = frac1
        b2, a2 = frac2
        a = math.lcm(a1, a2)
        b = b1 * (a // a1) + b2 * (a // a2)
        g = math.gcd(b, a)
        b //= g
        a //= g
        if a < 0:
            a *= -1
            b *= -1
        return [b, a]

    @staticmethod
    def frac_max(frac1: List[int], frac2: List[int]) -> List[int]:
        # Both denominators a1 and a2 are required to be non-zero
        b1, a1 = frac1
        b2, a2 = frac2
        if a1 < 0:
            a1 *= -1
            b1 *= -1
        if a2 < 0:
            a2 *= -1
            b2 *= -1
        if b1 * a2 < b2 * a1:
            return [b2, a2]
        return [b1, a1]

    @staticmethod
    def frac_min(frac1: List[int], frac2: List[int]) -> List[int]:
        # Both denominators a1 and a2 are required to be non-zero
        b1, a1 = frac1
        b2, a2 = frac2
        if a1 < 0:
            a1 *= -1
            b1 *= -1
        if a2 < 0:
            a2 *= -1
            b2 *= -1
        if b1 * a2 > b2 * a1:
            return [b2, a2]
        return [b1, a1]

    @staticmethod
    def frac_ceil(frac: List[int]) -> int:
        # Both denominators a1 and a2 are required to be non-zero
        b, a = frac
        return math.ceil(b / a)
import math



class LexicoGraphicalOrder:
    def __init__(self):
        return

    @staticmethod
    def get_kth_num(n, k):
        # find the k-th smallest digit in the dictionary order within the range of 1 to n
        def check():
            c = 0
            first = last = num
            while first <= n:
                c += min(last, n) - first + 1
                last = last * 10 + 9
                first *= 10
            return c

        assert k <= n
        num = 1
        k -= 1
        while k:
            cnt = check()
            if k >= cnt:
                num += 1
                k -= cnt
            else:
                num *= 10
                k -= 1
        return num

    def get_num_kth(self, n, num):
        # Find the dictionary order of the number num within the range of 1 to n
        x = str(num)
        low = 1
        high = n
        while low < high - 1:
            # Using bisection for reverse engineering
            mid = low + (high - low) // 2
            st = str(self.get_kth_num(n, mid))
            if st < x:
                low = mid
            elif st > x:
                high = mid
            else:
                return mid
        return low if str(self.get_kth_num(n, low)) == x else high

    @staticmethod
    def get_kth_subset(n, k):
        # The k-th smallest subset of set [1,...,n], with a total of 1<<n subsets
        assert k <= (1 << n)
        ans = []
        if k == 1:
            # Empty subset output 0
            ans.append(0)
        k -= 1
        for i in range(1, n + 1):
            if k == 0:
                break
            if k <= pow(2, n - i):
                ans.append(i)
                k -= 1
            else:
                k -= pow(2, n - i)
        return ans

    def get_subset_kth(self, n, lst):
        # Dictionary order of subsets lst of set [1,..., n]
        low = 1
        high = n
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset(n, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset(n, low) == lst else high

    @staticmethod
    def get_kth_subset_comb(n, m, k):
        # Select the k-th comb of m elements from the set [1,...,n] to arrange the selection
        assert k <= math.comb(n, m)
        nums = list(range(1, n + 1))
        ans = []
        while k and nums and len(ans) < m:
            length = len(nums)
            c = math.comb(length - 1, m - len(ans) - 1)
            if c >= k:
                ans.append(nums.pop(0))
            else:
                k -= c
                nums.pop(0)
        return ans

    def get_subset_comb_kth(self, n, m, lst):
        # The lexicographic order of selecting m elements in the set [1,...,n]

        low = 1
        high = math.comb(n, m)
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset_comb(n, m, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset_comb(n, m, low) == lst else high

    @staticmethod
    def get_kth_subset_perm(n, k):
        # Select the k-th perm of n elements from the set [1,...,n] to arrange the perm selection
        s = math.factorial(n)
        assert 1 <= k <= s
        nums = list(range(1, n + 1))
        ans = []
        while k and nums:
            single = s // len(nums)
            i = (k - 1) // single
            ans.append(nums.pop(i))
            k -= i * single
            s = single
        return ans

    def get_subset_perm_kth(self, n, lst):
        # Dictionary order of perm permutation LST for n elements selected from set [1,...,n]
        low = 1
        high = math.factorial(n)
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset_perm(n, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset_perm(n, low) == lst else high


class Permutation:
    def __init__(self):
        return

    @staticmethod
    def next_permutation(nums):
        n = len(nums)
        for i in range(n - 2, -1, -1):
            if nums[i] < nums[i + 1]:
                ind = i + 1
                for j in range(i + 2, n):
                    if nums[i] < nums[j] < nums[ind]:
                        ind = j
                nums[i], nums[ind] = nums[ind], nums[i]
                nums[i + 1:] = sorted(nums[i + 1:])
                return nums
        nums.reverse()
        return nums

    @staticmethod
    def prev_permutation(nums):
        n = len(nums)
        post = math.inf
        for i in range(n - 1, -1, -1):
            if nums[i] > post:
                ind = -1
                for j in range(i + 1, n):
                    if nums[j] < nums[i]:
                        if ind == -1:
                            ind = j
                        elif nums[ind] < nums[j]:
                            ind = j
                nums[i], nums[ind] = nums[ind], nums[i]
                nums[i + 1:] = sorted(nums[i + 1:], reverse=True)
                return nums
            else:
                post = nums[i]
        nums.reverse()
        return nums
from src.basis.binary_search.template import BinarySearch


class LinearBasis:
    def __init__(self, m=64):
        self.m = m
        self.basis = [0] * self.m
        self.cnt = self.count_diff_xor()
        self.tot = 1 << self.cnt
        self.num = 0
        self.zero = 0
        self.length = 0
        return

    def minimize(self, x):
        for i in range(self.m):
            if x >> i & 1:
                x ^= self.basis[i]
        return x

    def add(self, x):
        assert x <= (1 << self.m) - 1
        x = self.minimize(x)
        self.num += 1
        if x:
            self.length += 1
        self.zero = int(self.length < self.num)

        for i in range(self.m - 1, -1, -1):
            if x >> i & 1:
                for j in range(self.m):
                    if self.basis[j] >> i & 1:
                        self.basis[j] ^= x
                self.basis[i] = x
                self.cnt = self.count_diff_xor()
                self.tot = 1 << self.cnt
                return True
        return False

    def count_diff_xor(self):
        num = 0
        for i in range(self.m):
            if self.basis[i] > 0:
                num += 1
        return num

    def query_kth_xor(self, x):
        res = 0
        for i in range(self.m):
            if self.basis[i]:
                if x & 1:
                    res ^= self.basis[i]
                x >>= 1
        return res

    def query_xor_kth(self, num):
        bs = BinarySearch()

        def check(x):
            return self.query_kth_xor(x) <= num

        return bs.find_int_right(0, self.tot - 1, check)

    def query_max(self):
        return self.query_kth_xor(self.tot - 1)

    def query_min(self):
        # include empty subset
        return self.query_kth_xor(0)

class LinearBasisVector:
    def __init__(self, m):
        self.basis = [[0] * m for _ in range(m)]
        self.m = m
        return

    def add(self, lst):
        for i in range(self.m):
            if self.basis[i][i] and lst[i]:
                a, b = self.basis[i][i], lst[i]
                self.basis[i] = [x * b for x in self.basis[i]]
                lst = [x * a for x in lst]
                lst = [lst[j] - self.basis[i][j] for j in range(self.m)]
        for j in range(self.m):
            if lst[j]:
                self.basis[j] = lst[:]
                return True
        return Falsefrom functools import reduce
from operator import xor


class Nim:
    def __init__(self, lst):
        self.lst = lst
        return

    def gen_result(self):
        return reduce(xor, self.lst) != 0
import math
import random
from collections import Counter




class NumBase:
    def __init__(self):
        return

    @staticmethod
    def get_k_bin_of_n(n, k):
        """K-base calculation of integer n supports both positive and negative bases"""
        assert abs(k) >= 2  # in principle, requirements
        if n == 0:  # binary, ternary, hexadecimal, and negative bases
            return [0]
        if k == 0:
            return []
        pos = 1 if k > 0 else -1
        k = abs(k)
        lst = []  # 0123456789" + "".join(chr(i+ord("A")) for i in range(26))
        while n:
            lst.append(n % k)
            n //= k
            n *= pos
        lst.reverse()
        return lst

    @staticmethod
    def k_bin_to_ten(k, st: str) -> str:
        """convert k-base characters to decimal characters"""
        order = "0123456789" + "".join(chr(i + ord("a")) for i in range(26))
        ind = {w: i for i, w in enumerate(order)}
        m = len(st)
        ans = 0
        for i in range(m):
            ans *= k
            ans += ind[st[i]]
        return str(ans)

    def ten_to_k_bin(self, k, st: str) -> str:
        """convert 10 base characters to k base characters"""
        order = "0123456789" + "".join(chr(i + ord("a")) for i in range(26))
        lst = self.get_k_bin_of_n(int(st), k)
        return "".join(order[i] for i in lst)


class RomeToInt:
    def __init__(self):
        return

    @staticmethod
    def int_to_roman(num: int) -> str:
        lst = [['I', 1], ['IV', 4], ['V', 5], ['IX', 9], ['X', 10], ['XL', 40], ['L', 50], ['XC', 90], ['C', 100],
               ['CD', 400], ['D', 500], ['CM', 900], ['M', 1000]]
        n = len(lst)
        i = n - 1
        ans = ''
        while i >= 0:
            if num >= lst[i][1]:
                k = num // lst[i][1]
                ans += k * lst[i][0]
                num -= k * lst[i][1]
                if num == 0:
                    return ans
            else:
                i -= 1
        return ans

    @staticmethod
    def roman_to_int(s: str) -> int:
        dct = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900, 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100,
               'D': 500, 'M': 1000}
        ans = i = 0
        n = len(s)
        while i < n:
            if i + 1 < n and s[i:i + 2] in dct:
                ans += dct[s[i:i + 2]]
                i += 2
            else:
                ans += dct[s[i]]
                i += 1
        return ans


class PrimeJudge:
    def __init__(self):
        return

    @staticmethod
    def is_prime_speed(num):  # faster!
        """https://zhuanlan.zhihu.com/p/107300262"""
        assert num > 0
        if num == 1:
            return False
        if (num == 2) or (num == 3):
            return True
        if (num % 6 != 1) and (num % 6 != 5):
            return False
        for i in range(5, int(math.sqrt(num)) + 1, 6):
            if (num % i == 0) or (num % (i + 2) == 0):
                return False
        return True

    @staticmethod
    def is_prime_general(num):
        """general square complexity"""
        assert num > 0
        if num == 1:
            return False
        for i in range(2, min(int(math.sqrt(num)) + 2, num)):
            if num % i == 0:
                return False
        return True

    @staticmethod
    def is_prime_random(num):
        """random guess may not be right"""
        assert num > 0
        if num == 2:
            return True
        if num == 1 or not num ^ 1:
            return False
        for i in range(128):  # can be adjusted
            rand = random.randint(2, num - 1)
            if pow(rand, num - 1, num) != 1:
                return False  # must not be prime number
        return True  # still may be prime number


class NumFactor:
    def __init__(self):
        return

    @staticmethod
    def get_all_factor(num):  # faster when 1 <= num <= 10**6!
        """Obtain all factors of an integer, including 1 and itself"""
        assert num >= 1
        pre = []
        post = []
        for i in range(1, int(math.sqrt(num)) + 1):
            if num % i == 0:
                pre.append(i)
                if num // i != i:
                    post.append(num // i)
        return pre + post[::-1]

    @staticmethod
    def get_all_factor_square(primes, num):  # 1 <= num <= 10**9!
        """Obtain all square factors of an integer, including 1"""
        lst = []
        for p in primes:
            cnt = 0
            while num % p == 0:
                num //= p
                cnt += 1
            if cnt > 1:
                lst.append((p, cnt // 2))
        if int(num ** 0.5) ** 2 == num:
            lst.append((int(num ** 0.5), 1))
        pre = {1}
        for p, c in lst:
            for num in list(pre):
                for i in range(1, c + 1):
                    pre.add(num * p ** i)
        return sorted([x * x for x in pre])

    @staticmethod
    def get_prime_factor(num):  # faster when 1 <= num <= 10**6!
        """prime factor decomposition supports up to 10**12"""
        assert num >= 1
        ans = []
        j = 2
        while j * j <= num:
            if num % j == 0:
                c = 0
                while num % j == 0:
                    num //= j
                    c += 1
                ans.append((j, c))
            j += 1
        if num > 1:
            ans.append((num, 1))
        return ans

    @staticmethod
    def get_prime_with_pollard_rho(num):  # faster when 10**6 <= num <= (1 << 64)!
        """returns the prime factorization of n and the corresponding number of factors for larger number"""

        def pollard_rho(n):
            # randomly return a factor of n [1, 1 << 64]
            if n & 1 == 0:
                return 2
            if n % 3 == 0:
                return 3

            s = ((n - 1) & (1 - n)).bit_length() - 1
            d = n >> s
            for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
                p = pow(a, d, n)
                if p == 1 or p == n - 1 or a % n == 0:
                    continue
                for _ in range(s):
                    prev = p
                    p = (p * p) % n
                    if p == 1:
                        return math.gcd(prev - 1, n)
                    if p == n - 1:
                        break
                else:
                    for i in range(2, n):
                        x, y = i, (i * i + 1) % n
                        f = math.gcd(x - y, n)
                        while f == 1:
                            x, y = (x * x + 1) % n, (y * y + 1) % n
                            y = (y * y + 1) % n
                            f = math.gcd(x - y, n)
                        if f != n:
                            return f
            return n

        if num <= 1:
            return Counter()  # special case

        pr = dict()
        sub = dict()
        stack = [num]
        while stack:
            m = stack.pop()
            if m > 0:
                if m in pr:
                    continue
                pr[m] = pollard_rho(m)
                if m == pr[m]:
                    sub[m] = Counter([m])
                    continue
                stack.append(~m)
                stack.append(m // pr[m])
                stack.append(pr[m])
            else:
                m = ~m
                sub[m] = sub[m // pr[m]] + sub[pr[m]]
        return sub[num]

    def get_all_with_pollard_rho(self, num):  # faster when 10**6 <= num <= (1 << 64)!
        """returns the prime factorization of n and the corresponding number of factors for larger number"""
        assert num >= 1

        if num == 1:
            return [1]
        cnt = self.get_prime_with_pollard_rho(num)
        pre = [1]
        for p in cnt:
            nex = []
            for w in pre:
                cur = w * p
                for x in range(1, cnt[p] + 1):
                    nex.append(cur)
                    cur *= p
            pre.extend(nex)
        return sorted(pre)


class PrimeSieve:
    def __init__(self):
        return

    @staticmethod
    def eratosthenes_sieve(n):  # faster!
        """eratosthenes screening method returns prime numbers less than or equal to n"""
        primes = [True] * (n + 1)
        p = 2
        while p * p <= n:
            if primes[p]:
                for i in range(p * 2, n + 1, p):
                    primes[i] = False
            p += 1
        primes = [element for element in range(2, n + 1) if primes[element]]
        return primes

    @staticmethod
    def euler_sieve(n):
        """euler linear sieve prime number"""
        flag = [False for _ in range(n + 1)]
        prime_numbers = []
        for num in range(2, n + 1):
            if not flag[num]:
                prime_numbers.append(num)
            for prime in prime_numbers:
                if num * prime > n:
                    break
                flag[num * prime] = True
                if num % prime == 0:
                    break
        return prime_numbers


class NumTheory:
    def __init__(self):
        return

    @staticmethod
    def least_square_sum(n: int) -> int:
        """Four Squares Theorem Each number can be represented by the complete sum of squares of at most four numbers"""
        while n % 4 == 0:
            n //= 4
        if n % 8 == 7:
            return 4
        for i in range(n + 1):
            temp = i * i
            if temp <= n:
                if int((n - temp) ** 0.5) ** 2 + temp == n:
                    return 1 + (0 if temp == 0 else 1)
            else:
                break
        return 3

    @staticmethod
    def nth_super_ugly_number(n: int, primes) -> int:
        """calculate the nth ugly number that only contains the prime factor in primes"""
        dp = [1] * n  # note that this includes 1
        m = len(primes)
        points = [0] * m
        for i in range(1, n):
            nex = math.inf
            for j in range(m):
                if primes[j] * dp[points[j]] < nex:
                    nex = primes[j] * dp[points[j]]
            dp[i] = nex
            for j in range(m):
                if primes[j] * dp[points[j]] == nex:
                    points[j] += 1
        return dp[n - 1]


class EulerPhi:

    def __init__(self):
        return

    @staticmethod
    def euler_phi_with_prime_factor(n):  # faster!
        """the euler function returns the number of coprime with n that are less than or equal to n"""
        # Note that 1 and 1 are coprime, while prime numbers greater than 1 are not coprime with 1
        assert n >= 1
        if n <= 10 ** 6:
            lst = NumFactor().get_prime_factor(n)
        else:
            cnt = NumFactor().get_prime_with_pollard_rho(n)
            lst = [(p, cnt[p]) for p in cnt]
        ans = n
        for p, _ in lst:
            ans = ans // p * (p - 1)
        return ans

    @staticmethod
    def euler_phi_general(n):
        """the euler function returns the number of coprime with n that are less than or equal to n"""
        # Note that 1 and 1 are coprime, while prime numbers greater than 1 are not coprime with 1
        assert n >= 1
        ans = n
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                ans = ans // i * (i - 1)
                while n % i == 0:
                    n = n // i
        if n > 1:
            ans = ans // n * (n - 1)
        return int(ans)
import math
from functools import reduce


class PeiShuTheorem:
    def __init__(self):
        return

    @staticmethod
    def get_lst_gcd(lst):
        return reduce(math.gcd, lst)
from collections import defaultdict


class AllFactorCnt:
    def __init__(self, n):
        self.n = n
        self.all_factor_cnt = [0, 1] + [2 for _ in range(2, n + 1)]
        for i in range(2, self.n + 1):
            for j in range(i * i, self.n + 1, i):
                self.all_factor_cnt[j] += 2 if j != i * i else 1
        return


class PrimeFactor:
    def __init__(self, n):
        self.n = n
        # calculate the minimum prime factor for all numbers from 1 to self.n
        self.min_prime = [0] * (self.n + 1)
        self.min_prime[1] = 1
        # determine whether all numbers from 1 to self.n are prime numbers
        self.prime_factor = [[] for _ in range(self.n + 1)]
        self.prime_factor_cnt = [0]*(self.n+1)
        self.prime_factor_mi_cnt = [0] * (self.n + 1)
        # calculate all factors of all numbers from 1 to self.n, including 1 and the number itself
        self.all_factor = [[], [1]] + [[1, i] for i in range(2, self.n + 1)]
        self.euler_phi = list(range(self.n+1))
        self.build()

        return

    def build(self):

        # complexity is O(nlogn)
        for i in range(2, self.n + 1):
            if not self.min_prime[i]:
                self.min_prime[i] = i
                for j in range(i * i, self.n + 1, i):
                    if not self.min_prime[j]:
                        self.min_prime[j] = i

        for num in range(2, self.n + 1):
            pre = num // self.min_prime[num]
            self.prime_factor_cnt[num] = self.prime_factor_cnt[pre] + int(self.min_prime[num] != self.min_prime[pre])
            cur = num
            p = self.min_prime[cur]
            cnt = 0
            while cur % p == 0:
                cnt += 1
                cur //= p
            self.prime_factor_mi_cnt[num] = self.prime_factor_mi_cnt[cur] + cnt

        # complexity is O(nlogn)
        for num in range(2, self.n + 1):
            i = num
            phi = num
            while num > 1:
                p = self.min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                self.prime_factor[i].append((p, cnt))
                phi =  phi // p * (p - 1)
            self.euler_phi[i] = phi

        # complexity is O(nlogn)
        for i in range(2, self.n + 1):
            for j in range(i * i, self.n + 1, i):
                self.all_factor[j].append(i)
                if j > i * i:
                    self.all_factor[j].append(j // i)
        for i in range(self.n + 1):
            self.all_factor[i].sort()
        return

    def comb(self, a, b):
        # Use prime factor decomposition to solve the values of combinatorial mathematics
        # and prime factor decomposition O ((a+b) log (a+b))
        cnt = defaultdict(int)
        for i in range(1, a + 1):  # a!
            for num, y in self.prime_factor[i]:
                cnt[num] += y
        for i in range(1, b + 1):  # b!
            for num, y in self.prime_factor[i]:
                cnt[num] -= y
        for i in range(1, a - b + 1):  # (a-b)!
            for num, y in self.prime_factor[i]:
                cnt[num] -= y
        ans = 1
        for w in cnt:
            ans *= w ** cnt[w]
        return ans

    def get_prime_numbers(self):
        return [i for i in range(2, self.n + 1) if self.min_prime[i] == 0]
import heapq
from typing import List


class ScanLine:
    def __init__(self):
        return

    @staticmethod
    def get_sky_line(buildings: List[List[int]]) -> List[List[int]]:

        # scan_line提取建筑物轮廓
        events = []
        # 生成左右端点事件并sorting
        for left, right, height in buildings:
            # [x,y,h]分别为左右端点与高度
            events.append([left, -height, right])
            events.append([right, 0, 0])
        events.sort()

        # 初始化结果与前序高度
        res = [[0, 0]]
        stack = [[0, float('math.inf')]]
        for left, height, right in events:
            # 超出管辖范围的先出队
            while left >= stack[0][1]:
                heapq.heappop(stack)
            # |入备选天际线队列
            if height < 0:
                heapq.heappush(stack, [height, right])
            # 高度发生变化出现新的关键点
            if res[-1][1] != -stack[0][0]:
                res.append([left, -stack[0][0]])
        return res[1:]

    @staticmethod
    def get_rec_area(rectangles: List[List[int]]) -> int:

        # scan_line提取矩形x轴的端点并sorting（也可以取y轴的端点是一个意思）
        axis = set()
        # [x1,y1,x2,y2] 为左下角到右上角坐标
        for rec in rectangles:
            axis.add(rec[0])
            axis.add(rec[2])
        axis = sorted(list(axis))
        ans = 0
        n = len(axis)
        for i in range(n - 1):

            # brute_force两个点之间的宽度
            x1, x2 = axis[i], axis[i + 1]
            width = x2 - x1
            if not width:
                continue

            # 这里本质上是求一维区间的合并覆盖长度作为高度
            items = [[rec[1], rec[3]] for rec in rectangles if rec[0] < x2 and rec[2] > x1]
            items.sort(key=lambda x: [x[0], -x[1]])
            height = low = high = 0
            for y1, y2 in items:
                if y1 >= high:
                    height += high - low
                    low, high = y1, y2
                else:
                    high = high if high > y2 else y2
            height += high - low

            # 表示区[x1,x2]内的矩形覆盖高度为height
            ans += width * height
        return ans



class UnWeightedTree:
    def __init__(self, n):
        self.n = n
        self.point_head = [0] * (self.n + 1)
        self.edge_from = [0]
        self.edge_to = [0]
        self.edge_next = [0]
        self.edge_id = 1
        self.parent = [-1]
        self.order = 0
        self.start = [-1]
        self.end = [-1]
        self.parent = [-1]
        self.depth = [0]
        self.order_to_node = [-1]
        return

    def add_directed_edge(self, u, v):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.edge_from.append(u)
        self.edge_to.append(v)
        self.edge_next.append(self.point_head[u])
        self.point_head[u] = self.edge_id
        self.edge_id += 1
        return

    def add_undirected_edge(self, u, v):
        assert 0 <= u < self.n
        assert 0 <= v < self.n
        self.add_directed_edge(u, v)
        self.add_directed_edge(v, u)
        return

    def get_edge_ids(self, u):
        assert 0 <= u < self.n
        i = self.point_head[u]
        ans = []
        while i:
            ans.append(i)
            i = self.edge_next[i]
        return

    def dfs_order(self, root=0):

        self.order = 0
        # index is original node value is dfs self.order
        self.start = [-1] * self.n
        # index is original node value is the maximum subtree dfs self.order
        self.end = [-1] * self.n
        # index is original node and value is its self.parent
        self.parent = [-1] * self.n
        stack = [root]
        # self.depth of every original node
        self.depth = [0] * self.n
        # index is dfs self.order and value is original node
        self.order_to_node = [-1] * self.n
        while stack:
            i = stack.pop()
            if i >= 0:
                self.start[i] = self.order
                self.order_to_node[self.order] = i
                self.end[i] = self.order
                self.order += 1
                stack.append(~i)
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    # the self.order of son nodes can be assigned for lexicographical self.order
                    if j != self.parent[i]:
                        self.parent[j] = i
                        self.depth[j] = self.depth[i] + 1
                        stack.append(j)
                    ind = self.edge_next[ind]
            else:
                i = ~i
                if self.parent[i] != -1:
                    self.end[self.parent[i]] = self.end[i]

        return

    def heuristic_merge(self):
        ans = [0] * self.n
        sub = [None for _ in range(self.n)]
        index = list(range(self.n))
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        stack = [0]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != self.parent[i]:
                        self.parent[j] = i
                        self.depth[j] = self.depth[i] + 1
                        stack.append(j)
                    ind = self.edge_next[ind]
            else:
                i = ~i
                sub[index[i]] = {self.depth[i]: 1}
                ans[i] = self.depth[i]
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != self.parent[i]:
                        a, b = index[i], index[j]
                        if len(sub[a]) > len(sub[b]):
                            res = ans[i]
                            a, b = b, a
                        else:
                            res = ans[j]

                        for x in sub[a]:
                            sub[b][x] = sub[b].get(x, 0) + sub[a][x]
                            if (sub[b][x] > sub[b][res]) or (sub[b][x] == sub[b][res] and x < res):
                                res = x
                        sub[a] = None
                        ans[i] = res
                        index[i] = b
                    ind = self.edge_next[ind]

        return [ans[i] - self.depth[i] for i in range(self.n)]

    # class Graph(UnWeightedTree):
    def tree_dp(self, nums):
        ans = [0] * self.n
        parent = [-1] * self.n
        stack = [0]
        res = max(nums)
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                ind = self.point_head[i]
                while ind:
                    j = self.edge_to[ind]
                    if j != parent[i]:
                        parent[j] = i
                        stack.append(j)
                    ind = self.edge_next[ind]
            else:
                i = ~i
                ind = self.point_head[i]
                a = b = 0
                while ind:
                    j = self.edge_to[ind]
                    if j != parent[i]:
                        cur = ans[j]
                        if cur > a:
                            a, b = cur, a
                        elif cur > b:
                            b = cur
                    ind = self.edge_next[ind]
                res = max(res, a + b + nums[i])
                ans[i] = a + nums[i]
        return res

class DFS:
    def __init__(self):
        return

    @staticmethod
    def gen_bfs_order_iteration(dct, root=0):
        """template of dfs order for rooted tree"""
        n = len(dct)
        for i in range(n):
            # visit from small to large according to the number of child nodes
            dct[i].sort(reverse=True)  # which is not necessary

        order = 0
        # index is original node value is dfs order
        start = [-1] * n
        # index is original node value is the maximum subtree dfs order
        end = [-1] * n
        # index is original node and value is its parent
        parent = [-1] * n
        stack = [root]
        # depth of every original node
        depth = [0] * n
        # index is dfs order and value is original node
        order_to_node = [-1] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                start[i] = order
                order_to_node[order] = i
                end[i] = order
                order += 1
                stack.append(~i)
                for j in dct[i]:
                    # the order of son nodes can be assigned for lexicographical order
                    if j != parent[i]:
                        parent[j] = i
                        depth[j] = depth[i] + 1
                        stack.append(j)
            else:
                i = ~i
                if parent[i] != -1:
                    end[parent[i]] = end[i]

        return start, end


class DfsEulerOrder:
    def __init__(self, dct, root=0):
        """dfs and euler order of rooted tree which can be used for online point update and query subtree sum"""
        n = len(dct)
        for i in range(n):
            # visit from small to large according to the number of child nodes
            dct[i].sort(reverse=True)  # which is not necessary
        # index is original node value is dfs order
        self.start = [-1] * n
        # index is original node value is the maximum subtree dfs order
        self.end = [-1] * n
        # index is original node and value is its parent
        self.parent = [-1] * n
        # index is dfs order and value is original node
        self.order_to_node = [-1] * n
        # index is original node and value is its depth
        self.node_depth = [0] * n
        # index is dfs order and value is its depth
        self.order_depth = [0] * n
        # the order of original node visited in the total backtracking path
        self.euler_order = []
        # the pos of original node first appears in the euler order
        self.euler_in = [-1] * n
        # the pos of original node last appears in the euler order
        self.euler_out = [-1] * n  # 每个原始节点再欧拉序中最后出现的位置
        self.build(dct, root)
        return

    def build(self, dct, root):
        """build dfs order and euler order and relative math.info"""
        order = 0
        stack = [(root, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                self.euler_order.append(i)
                self.start[i] = order
                self.order_to_node[order] = i
                self.end[i] = order
                self.order_depth[order] = self.node_depth[i]
                order += 1
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        # the order of son nodes can be assigned for lexicographical order
                        self.parent[j] = i
                        self.node_depth[j] = self.node_depth[i] + 1
                        stack.append((j, i))
            else:
                i = ~i
                if i != root:
                    self.euler_order.append(self.parent[i])
                if self.parent[i] != -1:
                    self.end[self.parent[i]] = self.end[i]
        for i, num in enumerate(self.euler_order):
            # pos of euler order for every original node
            self.euler_out[num] = i
            if self.euler_in[num] == -1:
                self.euler_in[num] = i
        return
from collections import deque



class Node:
    __slots__ = 'son', 'fail', 'last', 'len', 'val'

    def __init__(self):
        self.son = {}
        self.fail = self.last = None
        self.len = 0
        self.val = math.inf


class AhoCorasick:
    def __init__(self):
        self.root = Node()

    def insert(self, word, cost):
        x = self.root
        for c in word:
            if c not in x.son:
                x.son[c] = Node()
            x = x.son[c]
        x.len = len(word)
        x.val = min(x.val, cost)

    def set_fail(self):
        q = deque()
        for x in self.root.son.values():
            x.fail = x.last = self.root
            q.append(x)
        while q:
            x = q.popleft()
            for c, son in x.son.items():
                p = x.fail
                while p and c not in p.son:
                    p = p.fail
                son.fail = p.son[c] if p else self.root
                son.last = son.fail if son.fail.len else son.fail.last
                q.append(son)

    def search(self, target):
        pos = [[] for _ in range(len(target))]
        x = self.root
        for i, c in enumerate(target):
            while x and c not in x.son:
                x = x.fail
            x = x.son[c] if x else self.root
            cur = x
            while cur:
                if cur.len:
                    pos[i - cur.len + 1].append(cur.val)
                cur = cur.last
        return pos


class AcAutomaton:
    def __init__(self, p):
        self.m = sum(len(t) for t in p)
        self.n = len(p)
        self.p = p
        self.tr = [[0] * 26 for _ in range(self.m + 1)]
        self.end = [0] * (self.m + 1)
        self.fail = [0] * (self.m + 1)
        self.cnt = 0
        for i, t in enumerate(self.p):
            self.insert(i + 1, t)
        self.set_fail()
        return

    def insert(self, i: int, word: str):
        x = 0
        for c in word:
            c = ord(c) - ord('a')
            if self.tr[x][c] == 0:
                self.cnt += 1
                self.tr[x][c] = self.cnt
            x = self.tr[x][c]
        self.end[i] = x

    def search(self, s):
        freq = [0] * (self.cnt + 1)
        x = 0
        for c in s:
            x = self.tr[x][ord(c) - ord('a')]
            freq[x] += 1

        rg = [[] for _ in range(self.cnt + 1)]
        for i in range(self.cnt + 1):
            rg[self.fail[i]].append(i)

        vis = [False] * (self.cnt + 1)
        st = [0]
        while st:
            x = st[-1]
            if not vis[x]:
                vis[x] = True
                for y in rg[x]:
                    st.append(y)
            else:
                st.pop()
                for y in rg[x]:
                    freq[x] += freq[y]

        res = [freq[self.end[i]] for i in range(1, self.n + 1)]
        return res

    def set_fail(self):
        q = deque([self.tr[0][i] for i in range(26) if self.tr[0][i]])
        while q:
            x = q.popleft()
            for i in range(26):
                if self.tr[x][i] == 0:
                    self.tr[x][i] = self.tr[self.fail[x]][i]
                else:
                    self.fail[self.tr[x][i]] = self.tr[self.fail[x]][i]
                    q.append(self.tr[x][i])
        return
class Node(object):
    def __init__(self, val=" ", left=None, right=None):
        if val[0] == "+" and len(val) >= 2:
            val = val[1:]
        self.val = val
        self.left = left
        self.right = right


class TreeExpression:
    def __init__(self):
        return

    def exp_tree(self, s: str) -> Node:

        try:
            int(s)
            # number rest only
            return Node(s)
        except ValueError as _:
            pass

        # case start with -(
        if len(s) >= 2 and s[0] == "-" and s[1] == "(":
            cnt = 0
            for i, w in enumerate(s):
                if w == "(":
                    cnt += 1
                elif w == ")":
                    cnt -= 1
                    if not cnt:
                        pre = s[1:i].replace("+", "-").replace("-", "+")
                        s = pre + s[i:]
                        break
        # case start with -
        neg = ""
        if s[0] == "-":
            neg = "-"
            s = s[1:]

        n = len(s)
        cnt = 0
        # 按照运算符号的优先级reverse_order|遍历字符串
        for i in range(n - 1, -1, -1):
            cnt += int(s[i] == ')') - int(s[i] == '(')
            if s[i] in ['+', '-'] and not cnt:
                return Node(s[i], self.exp_tree(neg + s[:i]), self.exp_tree(s[i + 1:]))

        # 注意是从后往前
        for i in range(n - 1, -1, -1):
            cnt += int(s[i] == ')') - int(s[i] == '(')
            if s[i] in ['*', '/'] and not cnt:
                return Node(s[i], self.exp_tree(neg + s[:i]), self.exp_tree(s[i + 1:]))

        # 注意是从前往后
        for i in range(n):
            cnt += int(s[i] == ')') - int(s[i] == '(')
            if s[i] in ['^'] and not cnt:  # 这里的 ^ 表示幂
                return Node(s[i], self.exp_tree(neg + s[:i]), self.exp_tree(s[i + 1:]))

        # 其余则是开头结尾为括号的情况
        return self.exp_tree(s[1:-1])

    def main_1175(self, s):

        # 按照前序、中序与后序变成前缀中缀与后缀表达式
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            dfs(node.right)
            pre.append(node.val)
            return

        ans = []
        root = self.exp_tree(s)
        pre = []
        dfs(root)
        while len(pre) > 1:
            ans.append(pre)
            n = len(pre)
            stack = []
            for i in range(n):
                if pre[i] in "+-*/^":
                    op = pre[i]
                    b = stack.pop()
                    a = stack.pop()
                    op = "//" if op == "/" else op
                    op = "**" if op == "^" else op
                    stack.append(str(eval(f"{a}{op}{b}")))
                    stack += pre[i + 1:]
                    break
                else:
                    stack.append(pre[i])
            pre = stack[:]
        ans.append(pre)
        return ans


class EnglishNumber:
    def __init__(self):
        return

    @staticmethod
    def number_to_english(n):

        # 将 0-9999 的数字转换为美式英语即有 and
        one = ["", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine",
               "ten", "eleven", "twelve", "thirteen", "fourteen",
               "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]

        ten = [
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety"]

        for word in ten:
            one.append(word)
            for i in range(1, 10):
                one.append(word + " " + one[i])

        ans = ""
        s = str(n)
        if n >= 1000:
            ans += one[n // 1000] + " thousand "

        if (n % 1000) // 100 > 0:
            ans += one[n % 1000 // 100] + " hundred "
        if (n >= 100 and 0 < n % 100 < 10) or (n >= 1000 and 0 < n % 1000 < 100):
            ans += "and "
        ans += one[n % 100]

        if ans == "":
            return "zero"
        return ans
class KMP:
    def __init__(self):
        return

    @classmethod
    def prefix_function(cls, s):
        """calculate the longest common true prefix and true suffix for s [:i+1] and s [:i+1]"""
        n = len(s)  # fail tree
        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:  # all pi[i] pi[pi[i]] ... are border
                j += 1   # all i+1-pi[i] pi[i]+1-pi[pi[i]] ... are circular_section
            pi[i] = j  # pi[i] <= i also known as next
        # pi[0] = 0
        return pi    # longest common true prefix_suffix / i+1-nex[i] is shortest circular_section

    @staticmethod
    def z_function(s):
        """calculate the longest common prefix between s[i:] and s"""
        n = len(s)
        z = [0] * n
        left, r = 0, 0
        for i in range(1, n):
            if i <= r and z[i - left] < r - i + 1:
                z[i] = z[i - left]
            else:
                z[i] = max(0, r - i + 1)
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
            if i + z[i] - 1 > r:
                left = i
                r = i + z[i] - 1
        # z[0] = 0
        return z

    def prefix_function_reverse(self, s):
        n = len(s)
        nxt = [0] + self.prefix_function(s)
        nxt[1] = 0
        for i in range(2, n + 1):
            j = i
            while nxt[j]:
                j = nxt[j]
            if nxt[i]:
                nxt[i] = j
        return nxt[1:]  # shortest common true prefix_suffix / i+1-nex[i] is longest circular_section

    def find(self, s1, s2):
        """find the index position of s2 in s1"""
        n, m = len(s1), len(s2)
        pi = self.prefix_function(s2 + "#" + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans

    def find_lst(self, s1, s2, tag=-1):
        """find the index position of s2 in s1"""
        n, m = len(s1), len(s2)
        pi = self.prefix_function(s2 + [tag] + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans

    def find_longest_palindrome(self, s, pos="prefix") -> int:
        """calculate the longest prefix and longest suffix palindrome substring"""
        if pos == "prefix":
            return self.prefix_function(s + "#" + s[::-1])[-1]
        return self.prefix_function(s[::-1] + "#" + s)[-1]

    @staticmethod
    def kmp_automaton(s, m=26):
        n = len(s)
        nxt = [0] * m * (n + 1)
        j = 0
        for i in range(1, n + 1):
            j = nxt[j * m + s[i - 1]]
            nxt[(i - 1) * m + s[i - 1]] = i
            for k in range(m):
                nxt[i * m + k] = nxt[j * m + k]
        return nxt

    @classmethod
    def merge_b_from_a(cls, a, b):
        c = b + "#" + a
        f = cls.prefix_function(c)
        m = len(b)
        if max(f[m:]) == m:
            return a
        x = f[-1]
        return a + b[x:]
class LyndonDecomposition:
    def __init__(self):
        return


    @staticmethod
    def solve_by_duval(s):
        """template of duval algorithm"""
        n, i = len(s), 0
        factorization = []
        while i < n:
            j, k = i + 1, i
            while j < n and s[k] <= s[j]:
                if s[k] < s[j]:
                    k = i
                else:
                    k += 1
                j += 1
            while i <= k:
                factorization.append(s[i: i + j - k])
                i += j - k
        return factorization

    @staticmethod
    def min_cyclic_string(s):
        """template of smallest cyclic string"""
        s += s
        n = len(s)
        i, ans = 0, 0
        while i < n // 2:
            ans = i
            j, k = i + 1, i
            while j < n and s[k] <= s[j]:
                if s[k] < s[j]:
                    k = i
                else:
                    k += 1
                j += 1
            while i <= k:
                i += j - k
        return s[ans: ans + n // 2]

    @staticmethod
    def min_express(sec):
        """template of minimum lexicographic expression"""
        n = len(sec)
        k, i, j = 0, 0, 1
        while k < n and i < n and j < n:
            if sec[(i + k) % n] == sec[(j + k) % n]:
                k += 1
            else:
                if sec[(i + k) % n] > sec[(j + k) % n]:
                    i = i + k + 1
                else:
                    j = j + k + 1
                if i == j:
                    i += 1
                k = 0
        i = i if i < j else j
        return i, sec[i:] + sec[:i]

    @staticmethod
    def max_express(sec):
        """template of maximum lexicographic expression"""
        n = len(sec)
        k, i, j = 0, 0, 1
        while k < n and i < n and j < n:
            if sec[(i + k) % n] == sec[(j + k) % n]:
                k += 1
            else:
                if sec[(i + k) % n] < sec[(j + k) % n]:
                    i = i + k + 1
                else:
                    j = j + k + 1
                if i == j:
                    i += 1
                k = 0
        i = i if i < j else j
        return i, sec[i:] + sec[:i]
class ManacherPlindrome:
    def __init__(self):
        return

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def manacher(s):
        """template of get the palindrome radius for every i-th character as center"""
        n = len(s)
        arm = [0] * n
        left, right = 0, -1
        for i in range(0, n):
            a, b = arm[left + right - i], right - i + 1
            a = a if a < b else b
            k = 0 if i > right else a
            while 0 <= i - k and i + k < n and s[i - k] == s[i + k]:
                k += 1
            arm[i] = k
            k -= 1
            if i + k > right:
                left = i - k
                right = i + k
        # s[i-arm[i]+1: i+arm[i]] is palindrome substring for every i
        return arm

    def palindrome_start_end(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        start = [[] for _ in range(n)]
        # the starting position index of the palindrome substring ending with the current index as the boundary
        end = [[] for _ in range(n)]
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    start[left // 2].append(right // 2)
                    end[right // 2].append(left // 2)
                left += 1
                right -= 1
        return start, end

    def palindrome_post_pre(self, s: str) -> (list, list):
        """template of get the length of the longest palindrome substring that starts or ends at a certain position"""
        n = len(s)
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)
        post = [1] * n
        pre = [1] * n
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    post[x] = max(post[x], y - x + 1)
                    pre[y] = max(pre[y], y - x + 1)
                    break
                left += 1
                right -= 1
        for i in range(1, n):
            if i - pre[i - 1] - 1 >= 0 and s[i] == s[i - pre[i - 1] - 1]:
                pre[i] = max(pre[i], pre[i - 1] + 2)
        for i in range(n - 2, -1, -1):
            pre[i] = max(pre[i], pre[i + 1] - 2)
        for i in range(n - 2, -1, -1):
            if i + post[i + 1] + 1 < n and s[i] == s[i + post[i + 1] + 1]:
                post[i] = max(post[i], post[i + 1] + 2)
        for i in range(1, n):
            post[i] = max(post[i], post[i - 1] - 2)

        return post, pre

    def palindrome_longest_length(self, s: str) -> (list, list):
        """template of get the longest palindrome substring of s"""
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)
        ans = 0
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            cur = (right - left + 1) // 2
            ans = ans if ans > cur else cur
        return ans

    def palindrome_just_start(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        start = []
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    if left // 2 == 0:
                        start.append(right // 2)
                    break
                left += 1
                right -= 1
        return start  # prefix palindrome

    def palindrome_just_end(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        end = []
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    if right // 2 == n-1:
                        end.append(left // 2)
                    break
                left += 1
                right -= 1
        return end  # suffix palindrome

    def palindrome_count_start_end(self, s: str) -> (list, list):
        """template of get the number of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        start = [0] * n
        end = [0] * n
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    if (y - x + 1) % 2:
                        mid = x + (y - x + 1) // 2
                        start[x] += 1
                        if mid + 1 < n:
                            start[mid + 1] -= 1
                        end[mid] += 1
                        if y + 1 < n:
                            end[y + 1] -= 1
                    else:
                        mid = x + (y - x + 1) // 2 - 1
                        start[x] += 1
                        start[mid + 1] -= 1
                        end[mid + 1] += 1
                        if y + 1 < n:
                            end[y + 1] -= 1
                    break
                left += 1
                right -= 1
        for i in range(1, n):
            start[i] += start[i - 1]
            end[i] += end[i - 1]
        return start, end

    def palindrome_count_start_end_odd(self, s: str) -> (list, list):
        """template of get the number of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        start = [0] * n
        end = [0] * n
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    if (y - x + 1) % 2:
                        mid = x + (y - x + 1) // 2
                        start[x] += 1
                        if mid + 1 < n:
                            start[mid + 1] -= 1
                        end[mid] += 1
                        if y + 1 < n:
                            end[y + 1] -= 1
                    break
                left += 1
                right -= 1
        for i in range(1, n):
            start[i] += start[i - 1]
            end[i] += end[i - 1]
        return start, end


    def palindrome_length_count(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        odd = [0] * (n + 2)
        even = [0] * (n + 2)

        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    if (y - x + 1) % 2:
                        low, high = 1, (y - x + 2) // 2
                        odd[low] += 1
                        if high + 1 <= n:
                            odd[high + 1] -= 1
                    else:
                        low, high = 1, (y - x + 1) // 2
                        even[low] += 1
                        if high + 1 <= n:
                            even[high + 1] -= 1
                    break
                left += 1
                right -= 1
        cnt = [0] * (n + 1)
        for i in range(1, n + 1):
            odd[i] += odd[i - 1]
            even[i] += even[i - 1]
            if 2 * i - 1 <= n:
                cnt[2 * i - 1] += odd[i]
            if 2 * i <= n:
                cnt[2 * i] += even[i]
        return cnt

    def palindrome_count(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        ans = 0

        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    if (y - x + 1) % 2:
                        ans += (y-x+2)//2
                    else:
                        ans += (y-x+1)//2
                    break
                left += 1
                right -= 1
        return ans
class PalindromeNum:
    def __init__(self):
        return

    @staticmethod
    def get_palindrome_num_1(n):
        """template of get all positive palindrome number with length not greater than n"""
        dp = [[""], [str(i) for i in range(10)]]
        for k in range(2, n + 1):
            # like dp to add palindrome character
            if k % 2 == 1:
                m = k // 2
                lst = []
                for st in dp[-1]:
                    for i in range(10):
                        lst.append(st[:m] + str(i) + st[m:])
                dp.append(lst)
            else:
                lst = []
                for st in dp[-2]:
                    for i in range(10):
                        lst.append(str(i) + st + str(i))
                dp.append(lst)

        nums = []
        for lst in dp:
            for num in lst:
                if num and num[0] != "0":
                    nums.append(int(num))
        nums.sort()
        return nums

    @staticmethod
    def get_palindrome_num_2(n):
        assert n >= 1
        """template of get all positive palindrome number whose length not greater than n"""
        nums = list(range(1, 10))
        x = 1
        while len(str(x)) * 2 <= n:
            num = str(x) + str(x)[::-1]
            nums.append(int(num))
            if len(str(x)) * 2 + 1 <= n:
                for d in range(10):
                    nums.append(int(str(x) + str(d) + str(x)[::-1]))
            x += 1
        nums.sort()
        return nums

    @staticmethod
    def get_palindrome_num_3():
        """template of get all positive palindrome number whose length not greater than n"""
        nums = list(range(10))
        for i in range(1, 10 ** 5):
            nums.append(int(str(i) + str(i)[::-1]))
            for j in range(10):
                nums.append(int(str(i) + str(j) + str(i)[::-1]))
        nums.sort()
        return nums

    @staticmethod
    def get_recent_palindrome_num(n: str) -> list:
        """template of recentest palindrome num of n"""
        m = len(n)
        candidates = [10 ** (m - 1) - 1, 10 ** m + 1]
        prefix = int(n[:(m + 1) // 2])
        for x in range(prefix - 1, prefix + 2):
            y = x if m % 2 == 0 else x // 10
            while y:
                x = x * 10 + y % 10
                y //= 10
            candidates.append(x)
        return candidates
import math
import random




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
        assert 0 <= x <= y <= self.n - 1
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
        base = max(max(lst) + 1, 150)
        self.p = random.randint(base, base * 2)
        self.mod = random.getrandbits(64)

        self.pre = [0] * (self.n + 1)
        self.pp = [1] * (self.n + 1)
        for j, w in enumerate(lst):
            self.pre[j + 1] = (self.pre[j] * self.p + w) % self.mod
            self.pp[j + 1] = (self.pp[j] * self.p) % self.mod
        return

    def query(self, x, y):
        """range hash value index start from 0"""
        assert 0 <= x <= y <= self.n - 1
        if y < x:
            return 0
        # with length y - x + 1 important!!!
        ans = (self.pre[y + 1] - self.pre[x] * self.pp[y - x + 1]) % self.mod
        return ans, y - x + 1

    def check(self, lst):
        ans = 0
        for w in lst:
            ans = (ans * self.p + w) % self.mod
        return ans, len(lst)


class PointSetRangeHashReverse:
    def __init__(self, n) -> None:
        self.n = n
        self.p = random.randint(26, 100)  # self.p = random.randint(150, 300)
        self.mod = random.randint(10 ** 9 + 7, (1 << 31) - 1)  # self.mod = random.getrandbits(64)
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


class StringHashSingleBuild:
    def __init__(self, n):
        """two mod to avoid hash crush or single 64-bit random mod"""
        self.n = n  # use two class to compute is faster!!!
        base = 150  # unique char or num in lst
        self.p = random.randint(base, base * 2)
        self.mod = random.getrandbits(64)
        self.pre = [0] * (self.n + 1)
        self.pp = [1] * (self.n + 1)
        for j in range(n):
            self.pp[j + 1] = (self.pp[j] * self.p) % self.mod
        return

    def build(self, lst):
        for j, w in enumerate(lst):
            self.pre[j + 1] = (self.pre[j] * self.p + w) % self.mod
        return

    def query(self, x, y):
        """range hash value index start from 0"""
        if not 0 <= x <= y <= self.n - 1:
            return 0
        ans = (self.pre[y + 1] - self.pre[x] * self.pp[y - x + 1]) % self.mod
        return ans  # with length y - x + 1 important!!!

class RangeSetRangeHashReverse:
    def __init__(self, n, tag=math.inf) -> None:
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
class SuffixArray:

    def __init__(self):
        return

    @staticmethod
    def build(s, sig):
        # sa: index is rank and value is pos
        # rk: index if pos and value is rank
        # height: lcp of rank i-th suffix and (i-1)-th suffix
        # sum(height): count of same substring of s
        # n*(n+1)//2 - sum(height): count of different substring of s
        # max(height): can compute the longest duplicate substring,
        # which is s[i: i + height[j]] and j = height.index(max(height)) and i = sa[j]
        # sig: number of unique rankings which initially is the size of the character set

        n = len(s)
        sa = list(range(n))
        rk = s[:]
        ll = 0  # ll is the length that has already been sorted, and now it needs to be sorted by 2ll length
        tmp = [0] * n
        while True:
            p = [i for i in range(n - ll, n)] + [x - ll for i, x in enumerate(sa) if x >= ll]
            # for suffixes with a length less than l, their second keyword ranking is definitely
            # the smallest because they are all empty
            # for suffixes of other lengths, suffixes starting at 'sa [i]' rank i-th, and their
            # first ll characters happen to be the second keyword of suffixes starting at 'sa[i] - ll'
            # start cardinality sorting, and first perform statistics on the first keyword
            # first, count how many values each has
            cnt = [0] * sig
            for i in range(n):
                cnt[rk[i]] += 1
            # make a prefix and for easy cardinality sorting
            for i in range(1, sig):
                cnt[i] += cnt[i - 1]

            # then use cardinality sorting to calculate the new sa
            for i in range(n - 1, -1, -1):
                w = rk[p[i]]
                cnt[w] -= 1
                sa[cnt[w]] = p[i]

            # new_sa to check new_rk
            def equal(ii, jj, lll):
                if rk[ii] != rk[jj]:
                    return False
                if ii + lll >= n and jj + lll >= n:
                    return True
                if ii + lll < n and jj + lll < n:
                    return rk[ii + lll] == rk[jj + lll]
                return False

            sig = -1
            for i in range(n):
                tmp[i] = 0

            for i in range(n):
                # compute the lcp
                if i == 0 or not equal(sa[i], sa[i - 1], ll):
                    sig += 1
                tmp[sa[i]] = sig

            for i in range(n):
                rk[i] = tmp[i]
            sig += 1
            if sig == n:
                break
            ll = ll << 1 if ll > 0 else 1

        # height
        k = 0
        height = [0] * n
        for i in range(n):
            if rk[i] > 0:
                j = sa[rk[i] - 1]
                while i + k < n and j + k < n and s[i + k] == s[j + k]:
                    k += 1
                height[rk[i]] = k
                k = 0 if k - 1 < 0 else k - 1
        return sa, rk, height
