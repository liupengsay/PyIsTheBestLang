
import unittest
from functools import lru_cache

from src.fast_io import FastIO

"""
算法：数位DP
功能：统计满足一定条件的自然数个数，也可以根据字典序大小特点统计一些特定字符串的个数，是一种计数常用的DP思想

题目：

===================================力扣===================================
233. 数字 1 的个数（https://leetcode.cn/problems/number-of-digit-one/）数字 1 的个数
357. 统计各位数字都不同的数字个数（https://leetcode.cn/problems/count-numbers-with-unique-digits/）排列组合也可用数位 DP 求解
600. 不含连续 1 的非负整数（https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/）不含连续 1 的非负整数
902. 最大为 N 的数字组合（https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/）限定字符情况下小于等于 n 的个数
1012. 至少有 1 位重复的数字（https://leetcode.cn/problems/numbers-with-repeated-digits/）容斥原理计算没有重复数字的个数
1067. 范围内的数字计数（https://leetcode.cn/problems/digit-count-in-range/）计算区间计数，使用右端点减去左端点，数位DP容斥模板题
1397. 找到所有好字符串（https://leetcode.cn/problems/find-all-good-strings/）使用数位DP思想进行模拟
2376. 统计特殊整数（https://leetcode.cn/problems/count-special-integers/）计算小于 n 的特殊正整数个数
2719. 统计整数数目（https://leetcode.cn/problems/count-of-integers/）数位DP容斥模板题
2801. 统计范围内的步进数字数目（https://leetcode.cn/problems/count-stepping-numbers-in-range/）数位DP容斥模板题
2827. 范围中美丽整数的数目（https://leetcode.cn/problems/number-of-beautiful-integers-in-the-range/）数位DP容斥模板题


面试题 17.06. 2出现的次数（https://leetcode.cn/problems/number-of-2s-in-range-lcci/）所有数位出现 2 的次数


===================================洛谷===================================
P1590 失踪的7（https://www.luogu.com.cn/problem/P1590）计算 n 以内不含7的个数
P1239 计数器（https://www.luogu.com.cn/problem/P1239）计算 n 以内每个数字0-9的个数
P3908 数列之异或（https://www.luogu.com.cn/problem/P3908）计算 1^2..^n的异或和，可以使用数位DP计数也可以用相邻的奇偶数计算
P1836 数页码（https://www.luogu.com.cn/problem/P1836）数位DP计算1~n内所有数字的数位和

参考：OI WiKi（xx）
"""


class DigitalDP:
    def __init__(self):
        return

    @staticmethod
    def count_bin(n):
        # 模板: 计算 1 到 n 的正整数二进制位 1 出现的次数
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
        ans = []  # 从二进制高位到二进制低位
        for w in range(m):
            cur = dfs(0, True, False, 0)
            ans.append(cur)
            dfs.cache_clear()
        return ans

    @staticmethod
    def count_digit(num, d):
        # 模板: 计算 1到 num 内数位 d 出现的个数
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
    def count_digit_iteration(num, d):
        # 模板: 计算 1到 num 内数位 d 出现的个数
        assert num >= 1
        s = str(num)
        n = len(s)
        dp = [[[[0] * 2 for _ in range(2)] for _ in range(n + 2)] for _ in range(n + 1)]
        # 数位 计数 是否受限 是否为数字
        for i in range(n, -1, -1):
            for cnt in range(n, -1, -1):
                for is_limit in range(1, -1, -1):
                    for is_num in range(1, -1, -1):
                        if i == n:
                            dp[i][cnt][is_limit][is_num] = cnt if is_num else 0
                            continue
                        res = 0
                        if not is_num:
                            res += dp[i + 1][0][0][0]
                        floor = 0 if is_num else 1
                        ceil = int(s[i]) if is_limit else 9
                        for x in range(floor, ceil + 1):
                            res += dp[i + 1][cnt + int(x == d)][int(is_limit and x == ceil)][1]
                        dp[i][cnt][is_limit][is_num] = res
        return dp[0][0][1][0]

    @staticmethod
    def count_num_base(num, d):
        # 模板: 使用进制计算 1 到 num 内不含数位 d 的数字个数
        assert 1 <= d <= 9  # 不含 0 则使用数位 DP 进行计算
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
            elif not d and lst[-1] == 0:  # 不含 0 应该怎么算
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

        # 模板: 使用进制计算 1 到 num 内不含数位 d 的数字个数
        assert 0 <= d <= 9

        @lru_cache(None)
        def dfs(i: int, is_limit: bool, is_num: bool) -> int:
            if i == m:
                return int(is_num)

            res = 0
            if not is_num:  # 可以跳过当前数位
                res = dfs(i + 1, False, False)
            up = int(s[i]) if is_limit else 9
            for x in range(0 if is_num else 1, up + 1):  # 枚举要填入的数字 d
                if x != d:
                    res += dfs(i + 1, is_limit and x == up, True)
            return res

        s = str(num)
        m = len(s)
        return dfs(0, True, False)

    @staticmethod
    def get_kth_without_d(k, d):
        # 模板: 使用进制计算第 k 个不含数位 d 的数 0<=d<=9
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
        # 也可以使用二分加数位 DP 进行求解
        ans = [str(st[i]) for i in lst]
        return int("".join(ans))


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_233(n: int) -> int:
        # 模板：计算 0 到 n 有数位 1 的出现次数
        if not n:
            return 0
        return DigitalDP().count_digit(n, 1)

    @staticmethod
    def lc_2719(num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        # 模板：数位DP容斥模板题

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
        # 模板：数位DP容斥模板题

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
        # 模板：数位DP容斥模板题

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
        # 模板：数位DP计算1~n内所有数字的数位和
        n = ac.read_int()
        ans = 0
        for d in range(1, 10):
            ans += d*DigitalDP().count_digit_iteration(n, d)
        ac.st(ans)
        return

    @staticmethod
    def lc_1067(d: int, low: int, high: int) -> int:
        # 模板：计算区间计数，使用右端点减去左端点，数位DP容斥模板题
        dd = DigitalDP()
        return dd.count_digit(high, d) - dd.count_digit(low-1, d)


class TestGeneral(unittest.TestCase):

    def test_digital_dp(self):

        dd = DigitalDP()
        cnt = [0] * 10
        n = 1000
        for i in range(1, n + 1):
            for w in str(i):
                cnt[int(w)] += 1

        for d in range(10):
            assert dd.count_digit(n, d) == cnt[d]
            assert dd.count_digit_iteration(n, d) == cnt[d]

        for d in range(1, 10):
            ans1 = dd.count_num_base(n, d)
            ans2 = sum(str(d) not in str(num) for num in range(1, n + 1))
            assert ans1 == ans2

        for d in range(10):
            ans1 = dd.count_num_dp(n, d)
            ans2 = sum(str(d) not in str(num) for num in range(1, n + 1))
            assert ans1 == ans2

        for d in range(10):
            nums = []
            for i in range(1, n + 1):
                if str(d) not in str(i):
                    nums.append(i)
            for i, num in enumerate(nums):
                assert dd.get_kth_without_d(i + 1, d) == num
        return


if __name__ == '__main__':
    unittest.main()
