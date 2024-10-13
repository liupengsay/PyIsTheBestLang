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
1355E（https://codeforces.com/problemset/problem/1355/E）ternary_search|classical|greedy
1389D（https://codeforces.com/problemset/problem/1389/D）ternary_search|brute_force|implemention|greedy
1374E2（https://codeforces.com/problemset/problem/1374/E2）ternary_search|two_pointers|brute_force|classical
1999G2（https://codeforces.com/problemset/problem/1999/G2）ternary_search|interactive|classical
578C（https://codeforces.com/problemset/problem/578/C）ternary_search|linear_dp|prefix_sum|classical

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
        dct_x = defaultdict(lambda: [math.inf, -inf])
        dct_y = defaultdict(lambda: [math.inf, -inf])
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

    @staticmethod
    def cf_1355e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1355/E
        tag: ternary_search|classical|greedy
        """
        n, a, r, m = ac.read_list_ints()
        h = ac.read_list_ints()

        h.sort()
        pre = ac.accumulate(h)

        def check(x):
            i = bisect.bisect_left(h, x)
            low = x * i - pre[i]
            high = pre[-1] - pre[i] - (n - i) * x
            y = min(low, high)
            res = y * min(m, a + r)
            res += (low - y) * a + (high - y) * r
            return res

        mid = TernarySearch().find_floor_point_int(check, 0, max(h))
        ans = min(check(x) for x in range(mid - 5, mid + 5) if x >= 0)
        ac.st(ans)
        return

    @staticmethod
    def cf_1389d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1389/D
        tag: ternary_search|brute_force|implemention|greedy
        """
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            l1, r1 = ac.read_list_ints()
            l2, r2 = ac.read_list_ints()

            if l1 > l2:
                l1, l2, r1, r2 = l2, l1, r2, r1

            def check(xx):
                gap = (l2 - r1) * xx
                cost = gap
                kk = k
                cur = min(xx * (r2 - l1), kk)
                kk -= cur
                cost += cur
                return cost + 2 * kk

            if r1 <= l2:
                x = TernarySearch().find_floor_point_int(check, 1, n)
                ans = min(check(y) for y in range(x - 5, x + 5) if 1 <= y <= n)
            else:
                zero = min(r1, r2) - max(l1, l2)
                one = r2 - l2 + r1 - l1 - 2 * zero
                one *= n
                zero *= n
                k -= min(k, zero)
                x = min(k, one)
                ans = x
                k -= x
                ans += k * 2
            ac.st(ans)
        return

    @staticmethod
    def cf_1374e2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1374/E2
        tag: ternary_search|two_pointers|brute_force|classical
        """
        n, m, k = ac.read_list_ints()
        aa = []
        bb = []
        cc = []
        dd = []
        for ii in range(n):
            t, a, b = ac.read_list_ints()
            if a == b == 1:
                cc.append((t, ii + 1))
            elif a:
                aa.append((t, ii + 1))
            elif b:
                bb.append((t, ii + 1))
            else:
                dd.append((t, ii + 1))
        ma = len(aa)
        mb = len(bb)
        mc = len(cc)
        md = len(dd)
        aa.sort()
        bb.sort()
        cc.sort()
        dd.sort()

        pre_a = ac.accumulate([a[0] for a in aa])
        pre_b = ac.accumulate([b[0] for b in bb])
        pre_c = ac.accumulate([c[0] for c in cc])

        aa.append((inf, 0))
        bb.append((inf, 0))
        dd.append((inf, 0))

        lst = []
        for xx in range(0, min(m, mc) + 1):
            val = xx
            rest = ma + mb + md
            if val < k:
                if ma < k - val or mb < k - val:
                    continue
                val += (k - val) * 2
                rest -= 2 * (k - val)
                if val > m:
                    continue
            if val + rest < m:
                continue
            lst.append(xx)
        if not lst:
            ac.st(-1)
            return

        def compute(x):
            cur = pre_c[x]
            cnt = x
            ia = ib = i = 0
            if x < k:
                cnt += 2 * (k - x)
                cur += pre_a[k - x] + pre_b[k - x]
                ia = ib = k - x
            for _ in range(m - cnt):
                if ia < ma and aa[ia][0] <= bb[ib][0] and aa[ia][0] <= dd[i][0]:
                    cur += aa[ia][0]
                    ia += 1
                elif ib < mb and bb[ib][0] <= aa[ia][0] and bb[ib][0] <= dd[i][0]:
                    cur += bb[ib][0]
                    ib += 1
                elif i < md and dd[i][0] <= bb[ib][0] and dd[i][0] <= aa[ia][0]:
                    cur += dd[i][0]
                    i += 1
            return cur, ia, ib, x, i

        def check(x):
            return compute(x)[0]

        y = TernarySearch().find_floor_point_int(check, lst[0], lst[-1])

        ans = []
        for yy in range(y - 5, y + 5):
            if lst[0] <= yy <= lst[-1]:
                res = compute(yy)
                if not ans or res < ans:
                    ans = res
        ac.st(ans[0])
        index = [a[1] for a in aa[:ans[1]]] + [b[1] for b in bb[:ans[2]]] + [c[1] for c in cc[:ans[3]]] + [d[1] for d in
                                                                                                           dd[:ans[4]]]
        ac.lst(index)
        return

    @staticmethod
    def cf_1999g2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1999/G2
        tag: ternary_search|interactive|classical
        """
        ac.flush = True
        for _ in range(ac.read_int()):
            left = 2
            right = 999
            while left < right - 2:
                diff = (right - left) // 3
                mid1 = left + diff
                mid2 = left + 2 * diff
                ac.lst(["?", mid1, mid2])
                res = ac.read_int()
                if res == mid1 * mid2:
                    left = mid2 + 1
                elif res == mid1 * (mid2 + 1):
                    left = mid1 + 1
                    right = mid2
                else:
                    right = mid1
            for x in range(left, right):
                ac.lst(["?", x, x])
                if ac.read_int() == (x + 1) * (x + 1):
                    ac.lst(["!", x])
                    break
            else:
                ac.lst(["!", right])
        return

    @staticmethod
    def cf_578c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/578/C
        tag: ternary_search|linear_dp|prefix_sum|classical
        """
        ac.read_int()
        nums = ac.read_list_ints()

        def check(x):
            high = -inf
            low = inf
            pre_high = pre_low = 0
            for num in nums:
                pre_high += num - x
                pre_low += num - x
                low = min(pre_low, low)
                high = max(pre_high, high)
                pre_high = max(pre_high, 0)
                pre_low = min(pre_low, 0)

            return max(abs(high), abs(low))

        ans = TernarySearch().find_floor_point_float(check, -10000, 10000, 1e-12)
        ac.st(check(ans))
        return
