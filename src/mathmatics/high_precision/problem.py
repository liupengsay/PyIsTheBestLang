"""
Algorithm：大数分解、素数判断、high_precision、分数代替浮点数运算
Function：xxx

====================================LeetCode====================================
166（https://leetcode.com/problems/fraction-to-recurring-decimal/）分数转换为有理数无限循环小数
172（https://leetcode.com/problems/factorial-trailing-zeroes/）阶乘后缀0的个数
1883（https://leetcode.com/problems/minimum-skips-to-arrive-at-meeting-on-time/description/）二维matrix_dp分数high_precision浮点数
2117（https://leetcode.com/problems/abbreviating-the-product-of-a-range/）大数或者prefix_suffiximplemention
972（https://leetcode.com/problems/equal-rational-numbers/）有理数转为分数判断

=====================================LuoGu======================================
2388（https://www.luogu.com.cn/problem/P2388）阶乘之乘后缀0的个数

1920（https://www.luogu.com.cn/problem/P1920）预估high_precision与公式 -ln(1-x) = sum(x**i/i for in range(1, n+1)) 其中 n 趋近于无穷
1729（https://www.luogu.com.cn/problem/P1729）high_precisione小数位
1727（https://www.luogu.com.cn/problem/P1727）high_precisionπ小数位
1517（https://www.luogu.com.cn/record/list?user=739032&status=12&page=5）high_precision小数的幂值
2394（https://www.luogu.com.cn/problem/P2394）high_precision
2393（https://www.luogu.com.cn/problem/P2393）high_precision

2399（https://www.luogu.com.cn/problem/P2399）小数有理数转换为最简分数
1530（https://www.luogu.com.cn/problem/P1530）分数化为小数

===================================CodeForces===================================
1144E（https://codeforces.com/contest/1144/problem/E）超大整数|减与乘除

====================================AtCoder=====================================
E - Double Factorial（https://atcoder.jp/contests/abc148/tasks/abc148_e）奇数阶乘与偶数阶乘的尾随零个数

"""

import math
from decimal import Decimal
from typing import List

from src.mathmatics.high_precision.template import HighPrecision, FloatToFrac
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_148e(ac=FastIO()):
        # 奇数阶乘与偶数阶乘的尾随零个数
        n = ac.read_int()
        if n % 2:
            ac.st(0)
        else:
            ans = HighPrecision().factorial_to_zero(n // 10) + n // 10
            ac.st(ans)
        return

    @staticmethod
    def cf_1144e(ac=FastIO()):
        # 超大整数|减与乘除
        n = ac.read_int()
        s = ac.read_str()
        t = ac.read_str()
        lst1 = [0] + [ord(w) - ord("a") for w in s]
        lst2 = [0] + [ord(w) - ord("a") for w in t]

        for i in range(n, 0, -1):
            lst1[i] += lst2[i]
            lst1[i - 1] += lst1[i] // 26
            lst1[i] %= 26

        for i in range(n + 1):
            rem = lst1[i] % 2
            lst1[i] //= 2
            if i + 1 <= n:
                lst1[i + 1] += rem * 26
            else:
                assert rem == 0

        ac.st("".join(chr(i + ord("a")) for i in lst1[1:]))
        return

    @staticmethod
    def lc_172(n):
        # 模板: n!的后缀零个数
        return HighPrecision().factorial_to_zero(n)

    @staticmethod
    def lc_972(s: str, t: str) -> bool:
        # 有理数转为分数判断
        hp = HighPrecision()
        return hp.decimal_to_fraction(s) == hp.decimal_to_fraction(t)

    @staticmethod
    def lg_p2238(ac=FastIO()):
        # 模板: 1!*2!*...*n!的后缀零个数
        n = ac.read_int()
        ac.st(HighPrecision().factorial_to_factorial(n))
        return

    @staticmethod
    def lg_p2399(ac=FastIO()):
        # 有理数转最简分数
        s = ac.read_str()
        a, b = HighPrecision().decimal_to_fraction(s)
        ac.st(f"{a}/{b}")
        return

    @staticmethod
    def lc_2217(left: int, right: int) -> str:
        # 大数或者prefix_suffiximplemention
        mod = 10 ** 20
        base = 10 ** 10
        zero = 0
        suffix = 1
        for x in range(left, right + 1):
            suffix *= x
            while suffix % 10 == 0:
                zero += 1
                suffix //= 10

            suffix %= mod

        prefix = 1
        for x in range(left, right + 1):
            prefix *= x
            while prefix % 10 == 0:
                prefix //= 10

            while prefix > mod:
                prefix //= 10

        if prefix >= base:
            return str(prefix)[:5] + "..." + str(suffix)[-5:] + "e" + str(zero)
        else:
            return str(prefix) + "e" + str(zero)

    @staticmethod
    def lg_p1530(ac=FastIO()):
        # 最简分数转化为有理数
        n, d = ac.read_list_ints()
        ans = HighPrecision().fraction_to_decimal(n, d)
        while ans:
            ac.st(ans[:76])
            ans = ans[76:]
        return

    @staticmethod
    def lc_1883_1(dist: List[int], speed: int, hours: int) -> int:

        # 二维matrix_dp分数high_precision浮点数
        n = len(dist)
        if sum(dist) > hours * speed:
            return -1

        ff = FloatToFrac()
        dp = [[[hours * 2, 1] for _ in range(n + 1)] for _ in range(n)]
        dp[0][0] = [0, 1]
        for i in range(n - 1):
            dp[i + 1][0] = ff.frac_add(dp[i][0], [ff.frac_ceil([dist[i], speed]), 1])
            for j in range(1, i + 2):
                pre1 = [ff.frac_ceil(ff.frac_add(dp[i][j], [dist[i], speed])), 1]
                pre2 = ff.frac_add(dp[i][j - 1], [dist[i], speed])
                dp[i + 1][j] = ff.frac_min(pre1, pre2)
        for j in range(n + 1):
            cur = ff.frac_add(dp[n - 1][j], [dist[n - 1], speed])
            if cur[0] <= hours * cur[1]:
                return j
        return -1

    @staticmethod
    def lc_1883_2(dist: List[int], speed: int, hours: int) -> int:

        # 二维matrix_dp分数high_precision浮点数
        cost = [Decimal(d) / Decimal(speed) for d in dist]
        n = len(dist)
        dp = [[hours * 2] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n):
            # 浮点数
            dp[i][0] = dp[i - 1][0] + math.ceil(cost[i - 1])
            for j in range(1, i):
                a, b = dp[i - 1][j - 1] + cost[i - 1], math.ceil(dp[i - 1][j] + cost[i - 1])
                dp[i][j] = a if a < b else b

            dp[i][i] = dp[i - 1][i - 1] + cost[i - 1]

        for j in range(n + 1):
            if dp[n - 1][j] + cost[-1] <= hours:
                return j
        return -1