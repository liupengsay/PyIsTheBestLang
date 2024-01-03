"""
Algorithm：digital_dp
Description：lexicographical_order|counter|high_to_low|low_to_high


====================================LeetCode====================================
233（https://leetcode.cn/problems/number-of-digit-one/）counter|digital_dp
357（https://leetcode.cn/problems/count-numbers-with-unique-digits/）comb|digital_dp
600（https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/）counter|digital_dp
902（https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/）counter|digital_dp
1012（https://leetcode.cn/problems/numbers-with-repeated-digits/）inclusion_exclusion|counter|digital_dp
1067（https://leetcode.cn/problems/digit-count-in-range/）counter|digital_dp|inclusion_exclusion
1397（https://leetcode.cn/problems/find-all-good-strings/）digital_dp|implemention
2376（https://leetcode.cn/problems/count-special-integers/）counter|digital_dp
2719（https://leetcode.cn/problems/count-of-integers/）digital_dp|inclusion_exclusion
2801（https://leetcode.cn/problems/count-stepping-numbers-in-range/）digital_dp|inclusion_exclusion
2827（https://leetcode.cn/problems/number-of-beautiful-integers-in-the-range/）digital_dp|inclusion_exclusion
17（https://leetcode.cn/problems/number-of-2s-in-range-lcci/）counter|digital_dp

====================================AtCoder=====================================
ABC121D（https://atcoder.jp/contests/abc121/tasks/abc121_d）xor_property|digital_dp
ABC208E（https://atcoder.jp/contests/abc208/tasks/abc208_e）brain_teaser|digital_dp

=====================================LuoGu======================================
P1590（https://www.luogu.com.cn/problem/P1590）counter|digital_dp
P1239（https://www.luogu.com.cn/problem/P1239）counter|digital_dp
P3908（https://www.luogu.com.cn/problem/P3908）xor_property|digital_dp|counter|odd_even
P1836（https://www.luogu.com.cn/problem/P1836）digital_dp

======================================Other======================================
（https://www.lanqiao.cn/problems/5891/learning/?contest_id=145）inclusion_exclusion|digital_dp

"""
from functools import lru_cache

from src.dp.digital_dp.template import DigitalDP
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_121d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc121/tasks/abc121_d
        tag: xor_property|digital_dp
        """
        #  n^(n+1) == 1 (n%2==0)
        def count(num):
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
                    res += dfs(i + 1, cnt + int(i == d and x == 1),
                               is_limit and ceil == x, True)
                return res

            if num <= 0:
                return 0
            s = bin(num)[2:]
            n = len(s)
            ans = 0
            for d in range(n):
                c = dfs(0, 0, True, False)
                dfs.cache_clear()
                if c % 2:
                    ans += 1 << (n - d - 1)
            return ans

        a, b = ac.read_list_ints()
        ac.st(count(b) ^ count(a - 1))
        return

    @staticmethod
    def abc_208e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc208/tasks/abc208_e
        tag: brain_teaser|digital_dp
        """
        @lru_cache(None)
        def dfs(i, is_limit, is_num, pre):
            if i == m:
                return int(is_num) and pre <= k
            res = 0
            if not is_num:
                res += dfs(i + 1, False, False, 0)
            low = 0 if is_num else 1
            high = int(st[i]) if is_limit else 9
            for x in range(low, high + 1):
                y = pre * x if is_num else x
                if y > k:
                    y = k + 1
                res += dfs(i + 1, is_limit and high == x, True, y)
            return res

        n, k = ac.read_list_ints()
        st = str(n)
        m = len(st)
        ans = dfs(0, True, False, 0)
        ac.st(ans)
        return

    @staticmethod
    def lc_233(n: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-digit-one/
        tag: counter|digital_dp
        """
        if not n:
            return 0
        return DigitalDP().count_digit(n, 1)

    @staticmethod
    def lc_2719(num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        """
        url: https://leetcode.cn/problems/count-of-integers/
        tag: digital_dp|inclusion_exclusion
        """
        def check(num):
            @lru_cache(None)
            def dfs(i, cnt, is_limit, is_num):
                if i == n:
                    if is_num:
                        return 1 if min_sum <= cnt <= max_sum else 0
                    return 0

                res = 0
                if not is_num:
                    res += dfs(i + 1, 0, False, False)
                floor = 0 if is_num else 1
                ceil = int(s[i]) if is_limit else 9
                for x in range(floor, ceil + 1):
                    if cnt + x <= max_sum:
                        res += dfs(i + 1, cnt + x, is_limit and ceil == x, True)
                    res %= mod
                return res

            s = str(num)
            n = len(s)
            ans = dfs(0, 0, True, False)
            dfs.cache_clear()
            return ans

        mod = 10 ** 9 + 7
        num2 = int(num2)
        num1 = int(num1)
        return (check(num2) - check(num1 - 1)) % mod

    @staticmethod
    def lc_2801(low: str, high: str) -> int:
        """
        url: https://leetcode.cn/problems/count-stepping-numbers-in-range/
        tag: digital_dp|inclusion_exclusion
        """
        def check(num):
            @lru_cache(None)
            def dfs(i, is_limit, is_num, pre):
                if i == n:
                    return is_num
                res = 0
                if not is_num:
                    res += dfs(i + 1, False, 0, -1)

                floor = 0 if is_num else 1
                ceil = int(s[i]) if is_limit else 9
                for x in range(floor, ceil + 1):
                    if pre == -1 or abs(x - pre) == 1:
                        res += dfs(i + 1, is_limit and ceil == x, 1, x)
                return res

            s = str(num)
            n = len(s)
            return dfs(0, True, 0, -1)

        mod = 10 ** 9 + 7
        return (check(int(high)) - check(int(low) - 1)) % mod

    @staticmethod
    def lc_2827(low: int, high: int, k: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-beautiful-integers-in-the-range/
        tag: digital_dp|inclusion_exclusion
        """
        def check(num):
            @lru_cache(None)
            def dfs(i, is_limit, is_num, odd, rest):
                if i == n:
                    return 1 if is_num and not odd and not rest else 0
                res = 0
                if not is_num:
                    res += dfs(i + 1, False, 0, 0, 0)

                floor = 0 if is_num else 1
                ceil = int(s[i]) if is_limit else 9
                for x in range(floor, ceil + 1):
                    res += dfs(i + 1, is_limit and ceil == x, 1, odd + 1 if x % 2 == 0 else odd - 1,
                               (rest * 10 + x) % k)
                return res

            s = str(num)
            n = len(s)
            return dfs(0, True, 0, 0, 0)

        return check(high) - check(low - 1)

    @staticmethod
    def lg_p1836(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1836
        tag: digital_dp
        """
        n = ac.read_int()
        ac.st(DigitalDP().count_digit_sum(n))
        return

    @staticmethod
    def lc_1067(d: int, low: int, high: int) -> int:
        """
        url: https://leetcode.cn/problems/digit-count-in-range/
        tag: counter|digital_dp|inclusion_exclusion
        """
        dd = DigitalDP()
        return dd.count_digit(high, d) - dd.count_digit(low - 1, d)
