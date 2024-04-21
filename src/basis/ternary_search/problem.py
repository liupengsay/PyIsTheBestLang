"""
Algorithm：ternary_search|TriPartPackTriPart|gradient_descent|hill_climbing
Description：used to find function extremum solutions with at most one peak or valley point in an interval

=====================================LuoGu======================================
1515（https://leetcode.cn/problems/best-position-for-a-service-centre/）tripartite_pack_tripartite|convex_function_minimum|gradient_descent|hill_climbing

=====================================LuoGu======================================
P3382（https://www.luogu.com.cn/problem/P3382）ternary_search|ceil
P1883（https://www.luogu.com.cn/problem/P1883）ternary_search|floor

===================================CodeForces===================================
939E（https://codeforces.com/problemset/problem/939/E）greedy|two_pointers|ternary_search|ceil
439D（https://codeforces.com/problemset/problem/439/D）ternary_search
1730B（https://codeforces.com/contest/1730/problem/B）ternary_search

====================================AtCoder=====================================
ABC130F（https://atcoder.jp/contests/abc130/tasks/abc130_f）ternary_search|floor|high_precision
ABC279D（https://atcoder.jp/contests/abc279/tasks/abc279_d）ternary_search|high_precision|classical
ABC240F（https://atcoder.jp/contests/abc240/tasks/abc240_f）implemention|ternary_search|binary_search|brute_force|classical


"""
import bisect
import math
import random
from collections import defaultdict
from decimal import Decimal
from typing import List

from src.basis.ternary_search.template import TernarySearch, TriPartPackTriPart
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_130f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc130/tasks/abc130_f
        tag: ternary_search|floor|high_precision
        """
        n = ac.read_int()
        ind = {"L": [-1, 0], "R": [1, 0], "U": [0, 1], "D": [0, -1]}
        dct_x = defaultdict(lambda: [inf, -inf])
        dct_y = defaultdict(lambda: [inf, -inf])
        for _ in range(n):
            x, y, d = ac.read_list_strs()
            x = int(x)
            y = int(y)

            a, b = dct_x[d]
            a = a if a < x else x
            b = b if b > x else x
            dct_x[d] = [a, b]

            a, b = dct_y[d]
            a = a if a < y else y
            b = b if b > y else y
            dct_y[d] = [a, b]

        lst_x = []
        for d in ind:
            if dct_x[d][0] < inf:
                for x in dct_x[d]:
                    lst_x.append([x, ind[d][0]])

        lst_y = []
        for d in ind:
            if dct_y[d][0] < inf:
                for y in dct_y[d]:
                    lst_y.append([y, ind[d][1]])

        def check(t):
            dis_x = [xx + t * aa for xx, aa in lst_x]
            dis_y = [xx + t * aa for xx, aa in lst_y]
            x_low = min(dis_x)
            x_high = max(dis_x)
            y_low = min(dis_y)
            y_high = max(dis_y)
            return (x_high - x_low) * (y_high - y_low)

        ans = TernarySearch().find_floor_value_float(check, 0, 10 ** 8, error=1e-10, high_precision=True)
        ac.st(ans)
        return

    @staticmethod
    def lg_1883(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1883
        tag: ternary_search|floor
        """
        t = ac.read_int()

        for _ in range(t):
            n = ac.read_int()
            nums = [ac.read_list_ints() for _ in range(n)]

            def fun(x):
                return max(a * x * x + b * x + c for a, b, c in nums)

            ans = TernarySearch().find_floor_value_float(fun, 0, 1000)
            ans = ac.round_5(ans * 10 ** 4)
            ans /= 10 ** 4
            ac.st("%.4f" % ans)
        return

    @staticmethod
    def lg_p3382(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3382
        tag: ternary_search|ceil
        """
        n, l, r = ac.read_list_floats()
        n = int(n)
        lst = ac.read_list_floats()
        lst.reverse()

        def check(x):
            res = lst[0]
            mul = 1
            for i in range(1, n + 1):
                mul *= x
                res += mul * lst[i]
            return res

        ans = TernarySearch().find_ceil_point_float(check, l, r)
        ac.st(ans)
        return

    @staticmethod
    def cf_439d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/439/D
        tag: ternary_search
        """
        n, m = ac.read_list_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        a.sort()
        b.sort()

        pre_a = [0] * (n + 1)
        for i in range(n):
            pre_a[i + 1] = pre_a[i] + a[i]

        pre_b = [0] * (m + 1)
        for i in range(m):
            pre_b[i + 1] = pre_b[i] + b[i]

        floor = min(a)
        ceil = max(b)
        if floor >= ceil:
            ac.st(0)
            return

        def check(xx):
            ii = bisect.bisect_right(a, xx)
            cost = ii * xx - pre_a[ii]

            ii = bisect.bisect_left(b, xx)
            cost += pre_b[-1] - pre_b[ii] - (m - ii) * xx
            return cost

        point = int(TernarySearch().find_floor_point_float(check, floor, ceil, 1))

        ans = inf
        for x in [-1, 0, 1]:
            if floor <= point + x <= ceil:
                ans = ac.min(ans, check(point + x))
        ac.st(ans)
        return

    @staticmethod
    def cf_939e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/939/E
        tag: greedy|two_pointers|ternary_search|ceil
        """
        nums = []
        pre = [0]

        def check(xx):
            return nums[-1] - (pre[xx] + nums[-1]) / (xx + 1)

        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if len(lst) == 2:
                nums.append(lst[1])
                pre.append(pre[-1] + nums[-1])
            else:
                n = len(nums)
                x = TernarySearch().find_ceil_point_int(check, 0, n - 1)
                ans = -inf
                for y in [x - 1, x, x + 1, x + 2]:
                    if 0 <= y <= n - 1:
                        ans = ac.max(ans, check(y))
                ac.st(ans)
        return

    @staticmethod
    def cf_1730b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1730/problem/B
        tag: ternary_search
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            x = ac.read_list_ints()
            t = ac.read_list_ints()

            def check(x0):
                return max(t[i] + abs(x[i] - x0) for i in range(n))

            ans = TernarySearch().find_floor_point_float(check, min(x), max(x), 1e-8)
            ac.st(ans)
        return

    @staticmethod
    def lc_1515_1(stack: List[List[int]]) -> float:
        """
        url: https://leetcode.cn/problems/best-position-for-a-service-centre/
        tag: tripartite_pack_tripartite|convex_function_minimum|gradient_descent|hill_climbing
        """

        def target(x, y):
            return sum([math.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2) for p in stack])

        low_x = min([p[0] for p in stack])
        high_x = max([p[0] for p in stack])
        low_y = min([p[1] for p in stack])
        high_y = max([p[1] for p in stack])
        _, _, ans = TriPartPackTriPart().find_floor_point_float(target, low_x, high_x, low_y, high_y)
        return ans ** 2

    @staticmethod
    def lc_1515_2(positions: List[List[int]]) -> float:
        """
        url: https://leetcode.cn/problems/best-position-for-a-service-centre/
        tag: tripartite_pack_tripartite|convex_function_minimum|gradient_descent|hill_climbing
        """
        eps = 1e-10
        alpha = 1.0
        decay = 0.001
        n = len(positions)
        batch_size = n
        x = sum(pos[0] for pos in positions) / n
        y = sum(pos[1] for pos in positions) / n

        while True:
            x_pre, y_pre = x, y
            random.shuffle(positions)
            for i in range(0, n, batch_size):
                dx = dy = 0.0
                j = i + batch_size if i + batch_size < n else n
                for k in range(i, j):
                    pos = positions[k]
                    dx += (x - pos[0]) / ((x - pos[0]) * (x - pos[0]) + (y - pos[1]) * (y - pos[1]) + eps) ** 0.5
                    dy += (y - pos[1]) / ((x - pos[0]) * (x - pos[0]) + (y - pos[1]) * (y - pos[1]) + eps) ** 0.5
                x -= alpha * dx
                y -= alpha * dy
            alpha *= (1 - decay)
            if ((x - x_pre) * (x - x_pre) + (y - y_pre) * (y - y_pre)) ** 0.5 < eps:
                break

        ans = sum(((x - x0) * (x - x0) + (y - y0) * (y - y0)) ** 0.5 for x0, y0 in positions)
        return ans

    @staticmethod
    def lc_1515_3(positions: List[List[int]]) -> float:
        """
        url: https://leetcode.cn/problems/best-position-for-a-service-centre/
        tag: tripartite_pack_tripartite|convex_function_minimum|gradient_descent|hill_climbing
        """
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        eps = 1e-6
        step = 1.0
        decay = 0.8
        n = len(positions)

        x = sum(pos[0] for pos in positions) / n
        y = sum(pos[1] for pos in positions) / n

        def get_dis(xc, yc):
            return sum(((xc - x0) * (xc - x0) + (yc - y0) * (yc - y0)) ** 0.5 for x0, y0 in positions)

        while step > eps:
            for dx, dy in dirs:
                x_next = x + step * dx
                y_next = y + step * dy
                if get_dis(x_next, y_next) < get_dis(x, y):
                    x, y = x_next, y_next
                    break
            else:
                step *= (1.0 - decay)
        return get_dis(x, y)

    @staticmethod
    def abc_279d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc279/tasks/abc279_d
        tag: ternary_search|high_precision|classical
        """
        a, b = ac.read_list_ints()

        def check(x):
            if x < 0:
                return inf
            return x * b + a / (1 + x) ** 0.5

        y = TernarySearch().find_floor_point_int(check, 0, a)
        y = int(y)
        ans = min(Decimal(check(x)) for x in range(y - 5, y + 6))
        ac.st(ans)
        return

    @staticmethod
    def abc_240f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc240/tasks/abc240_f
        tag: implemention|ternary_search|binary_search|brute_force|classical
        """
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            nums = [ac.read_list_ints() for _ in range(n)]
            ans = nums[0][0]
            pre = 0
            pre_pre = 0

            def check(s):
                return pre_pre + pre * s + s * (s + 1) * x // 2

            for x, y in nums:
                ans = max(ans, pre_pre + pre + x)
                if x < 0:
                    ceil = TernarySearch().find_ceil_point_int(check, 1, y)
                    for ss in range(ceil - 5, ceil + 5):
                        if 1 <= ss <= y:
                            ans = max(ans, check(ss))
                pre_pre = pre_pre + pre * y + y * (y + 1) * x // 2
                pre += x * y
                ans = max(ans, pre_pre)
            ac.st(ans)
        return