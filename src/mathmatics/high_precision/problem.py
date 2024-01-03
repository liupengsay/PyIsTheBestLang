"""
Algorithm：big_number_factorization|is_prime|high_precision|float_to_frac|frac_to_float
Description：xxx

====================================LeetCode====================================
166（https://leetcode.cn/problems/fraction-to-recurring-decimal/）frac_to_float
172（https://leetcode.cn/problems/factorial-trailing-zeroes/）suffix_zero|factorial
1883（https://leetcode.cn/problems/minimum-skips-to-arrive-at-meeting-on-time/）matrix_dp|high_precision|float_to_frac
2117（https://leetcode.cn/problems/abbreviating-the-product-of-a-range/）prefix_suffix|implemention
972（https://leetcode.cn/problems/equal-rational-numbers/）float_to_frac

=====================================LuoGu======================================
P2388（https://www.luogu.com.cn/problem/P2388）suffix_zero|factorial_of_factorial

P1920（https://www.luogu.com.cn/problem/P1920）high_precision|math
P1729（https://www.luogu.com.cn/problem/P1729）high_precision|e|math
P1727（https://www.luogu.com.cn/problem/P1727）high_precision|π|math
P1517（https://www.luogu.com.cn/problem/P1517）high_precision|float_power
P2394（https://www.luogu.com.cn/problem/P2394）high_precision
P2393（https://www.luogu.com.cn/problem/P2393）high_precision

P2399（https://www.luogu.com.cn/problem/P2399）float_to_frac
P1530（https://www.luogu.com.cn/problem/P1530）frac_to_float

===================================CodeForces===================================
1144E（https://codeforces.com/contest/1144/problem/E）big_number|minus|mul|divide

====================================AtCoder=====================================
ABC148E（https://atcoder.jp/contests/abc148/tasks/abc148_e）suffix_zero|odd_even|factorial

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
        """
        url: https://codeforces.com/contest/1144/problem/E
        tag: big_number|minus|mul|divide
        """
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
        """
        url: https://leetcode.cn/problems/factorial-trailing-zeroes/
        tag: suffix_zero|factorial
        """
        # 模板: n!的后缀零个数
        return HighPrecision().factorial_to_zero(n)

    @staticmethod
    def lc_972(s: str, t: str) -> bool:
        """
        url: https://leetcode.cn/problems/equal-rational-numbers/
        tag: float_to_frac
        """
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
        """
        url: https://www.luogu.com.cn/problem/P2399
        tag: float_to_frac
        """
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
        """
        url: https://www.luogu.com.cn/problem/P1530
        tag: frac_to_float
        """
        # 最简分数转化为有理数
        n, d = ac.read_list_ints()
        ans = HighPrecision().fraction_to_decimal(n, d)
        while ans:
            ac.st(ans[:76])
            ans = ans[76:]
        return

    @staticmethod
    def lc_1883_1(dist: List[int], speed: int, hours: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-skips-to-arrive-at-meeting-on-time/
        tag: matrix_dp|high_precision|float_to_frac
        """
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
        """
        url: https://leetcode.cn/problems/minimum-skips-to-arrive-at-meeting-on-time/
        tag: matrix_dp|high_precision|float_to_frac
        """
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
