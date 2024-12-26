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
P1298（https://www.luogu.com.cn/problem/P1298）high_precision|frac_to_float|brute_force

===================================CodeForces===================================
1144E（https://codeforces.com/contest/1144/problem/E）big_number|minus|mul|divide|high_precision|classical
1883E（https://codeforces.com/contest/1883/problem/E）high_precision|big_number|math|log
1995D（https://codeforces.com/contest/1995/problem/C）high_precision|greedy|implemention
1543C（https://codeforces.com/contest/1543/problem/C）high_precision|prob|expectation|implemention

====================================AtCoder=====================================
ABC148E（https://atcoder.jp/contests/abc148/tasks/abc148_e）suffix_zero|odd_even|factorial
ABC189D（https://atcoder.jp/contests/abc189/tasks/abc189_b）high_precision|division_to_multiplication
ABC191D（https://atcoder.jp/contests/abc191/tasks/abc191_d）high_precision|division_to_multiplication|brute_force
ABC385F（https://atcoder.jp/contests/abc385/tasks/abc385_f）math|slope|high_precision|monotonic_stack

====================================AtCoder=====================================
1（https://judge.yosupo.jp/problem/many_aplusb）big_number|high_precision|plus

"""
import decimal
import math
from decimal import Decimal
from typing import List

from src.math.high_precision.template import HighPrecision, FloatToFrac
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_148e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc148/tasks/abc148_e
        tag: suffix_zero|odd_even|factorial
        """

        n = ac.read_int()
        if n % 2:
            ac.st(0)
        else:
            ans = HighPrecision().factorial_suffix_zero_cnt(n // 10) + n // 10
            ac.st(ans)
        return

    @staticmethod
    def cf_1144e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1144/problem/E
        tag: big_number|minus|mul|divide|n_base
        """

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
        return HighPrecision().factorial_suffix_zero_cnt(n)

    @staticmethod
    def lc_972(s: str, t: str) -> bool:
        """
        url: https://leetcode.cn/problems/equal-rational-numbers/
        tag: float_to_frac
        """
        hp = HighPrecision()
        return hp.decimal_to_fraction(s) == hp.decimal_to_fraction(t)

    @staticmethod
    def lg_p2338(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2388
        tag: suffix_zero|factorial_of_factorial
        """
        n = ac.read_int()
        ac.st(HighPrecision().factorial_factorial_suffix_zero_cnt(n))
        return

    @staticmethod
    def lg_p2399(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2399
        tag: float_to_frac
        """
        s = ac.read_str()
        a, b = HighPrecision().decimal_to_fraction(s)
        ac.st(f"{a}/{b}")
        return

    @staticmethod
    def lc_2117(left: int, right: int) -> str:
        """
        url: https://leetcode.cn/problems/abbreviating-the-product-of-a-range/
        tag: prefix_suffix|implemention
        """
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

        cost = [Decimal(d) / Decimal(speed) for d in dist]
        n = len(dist)
        dp = [[hours * 2] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n):
            dp[i][0] = dp[i - 1][0] + math.ceil(cost[i - 1])
            for j in range(1, i):
                a, b = dp[i - 1][j - 1] + cost[i - 1], math.ceil(dp[i - 1][j] + cost[i - 1])
                dp[i][j] = a if a < b else b

            dp[i][i] = dp[i - 1][i - 1] + cost[i - 1]

        for j in range(n + 1):
            if dp[n - 1][j] + cost[-1] <= hours:
                return j
        return -1

    @staticmethod
    def abc_191d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc191/tasks/abc191_d
        tag: high_precision|division_to_multiplication|brute_force
        """
        m = 10000

        def check(s):
            if "." not in s:
                return int(s) * m
            while len(s) - s.index(".") - 1 < 4:
                s += "0"
            return int(s.replace(".", ""))

        x, y, r = [check(x) for x in ac.read_list_strs()]
        ans = 0
        low_x = (x - r) // m
        high_x = (x + r) // m
        for x0 in range(low_x, high_x + 1):
            ceil = r * r - (x - x0 * m) * (x - x0 * m)
            if ceil < 0:
                continue
            low = math.ceil(y - math.sqrt(ceil))
            high = math.floor(y + math.sqrt(ceil))
            low = low // m - 10
            high = high // m + 10
            while (low * m - y) * (low * m - y) + (x - x0 * m) * (x - x0 * m) > r * r and low <= high:
                low += 1
            while (high * m - y) * (high * m - y) + (x - x0 * m) * (x - x0 * m) > r * r and high >= low:
                high -= 1
            ans += high - low + 1
        ac.st(ans)
        return

    @staticmethod
    def abc_385f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc385/tasks/abc385_f
        tag: math|slope|high_precision|monotonic_stack
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        stack = [nums[0]]
        ans = -1
        for i in range(1, n):
            x, y = nums[i]
            while len(stack) >= 2 and (y - stack[-1][1]) * (x - stack[-2][0]) >= (x - stack[-1][0]) * (
                    y - stack[-2][1]):
                stack.pop()

            k = decimal.Decimal(y - stack[-1][1]) / decimal.Decimal(x - stack[-1][0])
            b = decimal.Decimal(y - k * x)
            ans = max(ans, b)
            stack.append((x, y))
        ac.st(ans if ans >= 0 else -1)
        return
