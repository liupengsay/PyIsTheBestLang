import unittest

from functools import lru_cache

from src.fast_io import FastIO
from src.mathmatics.number_theory import NumberTheory

"""
算法：博弈类DP、玩游戏、必胜态、必输态
功能：通常使用枚举、区间DP加模拟贪心的方式，和记忆化搜索进行状态转移
题目：

===================================力扣===================================
375. 猜数字大小 II（https://leetcode.cn/problems/guess-number-higher-or-lower-ii/）使用区间DP求解的典型博弈DP

===================================洛谷===================================
P1290 欧几里德的游戏（https://www.luogu.com.cn/problem/P1290）典型的博弈DP题
P5635 【CSGRound1】天下第一（https://www.luogu.com.cn/problem/P5635）博弈DP模拟与手写记忆化搜索，避免陷入死循环
P3150 pb的游戏（1）（https://www.luogu.com.cn/problem/P3150）博弈分析必胜策略与最优选择，只跟奇数偶数有关
P4702 取石子（https://www.luogu.com.cn/problem/P4702）博弈分析必胜策略与最优选择，只跟奇数偶数有关
P1247 取火柴游戏（https://www.luogu.com.cn/problem/P1247）nim博弈，使用异或求解
P1512 伊甸园日历游戏（https://www.luogu.com.cn/problem/P1512）博弈DP与日期操作
P2092 数字游戏（https://www.luogu.com.cn/problem/P2092）根据质数的个数来判断必胜态
P2953 [USACO09OPEN]Cow Digit Game S（https://www.luogu.com.cn/problem/P2953）必胜态线性DP

参考：OI WiKi（xx）
"""

class DateTime:
    def __init__(self):
        self.leap_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.not_leap_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return

    def is_leap_year(self, yy):
        # 模板: 判断是否为闰年
        assert sum(self.leap_month) == 366
        assert sum(self.not_leap_month) == 365
        # 闰年天数
        return yy % 400 == 0 or (yy % 4 == 0 and yy % 100 != 0)

    def year_month_day_cnt(self, yy, mm):
        ans = self.leap_month[mm-1] if self.is_leap_year(yy) else self.not_leap_month[mm-1]
        return ans

    def is_valid(self, yy, mm, dd):
        if not [1900, 1, 1] <= [yy, mm, dd] <= [2006, 11, 4]:
            return False
        day = self.year_month_day_cnt(yy, mm)
        if not 1<=dd<=day:
            return False
        return True


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1280(ac=FastIO()):
        # 模板：博弈 DP 下的必胜策略分析
        n = ac.read_int()
        for _ in range(n):
            x, y = ac.read_ints()

            @lru_cache(None)
            def dfs(a, b):
                if a < b:
                    a, b = b, a
                if a % b == 0:
                    return True
                if a//b >= 2:  # 注意分类贪心进行必胜态考量
                    return True
                for i in range(1, a // b + 1):
                    if not dfs(a - i * b, b):
                        return True
                return False

            ans = dfs(x, y)
            if ans:
                ac.st("Stan wins")
            else:
                ac.st("Ollie wins")
        return

    @staticmethod
    def lg_1247(ac=FastIO()):
        # 模板：nim博弈，使用异或求解
        k = ac.read_int()
        nums = ac.read_list_ints()
        x = reduce(xor, nums)
        if x == 0:
            ac.st("lose")
        else:
            for i in range(k):
                if nums[i] ^ x < nums[i]:
                    res = [nums[i] - (nums[i] ^ x), i+1]
                    ac.lst(res)
                    nums[i] = x ^ nums[i]
                    ac.lst(nums)
                    return
        return

    @staticmethod
    def lg_p1512(ac=FastIO()):
        # 模板：博弈DP与日期操作
        dt = DateTime()
        stack = [[1900, 1, 1]]
        yy, mm, dd = stack[0]
        dates = []
        while [yy, mm, dd] < [2006, 11, 4]:
            if dd + 1 <= dt.year_month_day_cnt(yy, mm):
                cur = [yy, mm, dd + 1]
            elif mm + 1 <= 12:
                cur = [yy, mm + 1, 1]
            else:
                cur = [yy + 1, 1, 1]
            yy, mm, dd = cur
            dates.append(cur)
        dct = set(tuple(dt) for dt in dates)

        dp = dict()
        dp[(2006, 11, 4)] = False
        for yy, mm, dd in dates[:-1][::-1]:
            dp[(yy, mm, dd)] = False

            cur = [yy, mm + 1, dd] if mm + 1 <= 12 else [yy + 1, 1, dd]
            if (cur[0], cur[1], cur[2]) in dct and not dp[(cur[0], cur[1], cur[2])]:
                dp[(yy, mm, dd)] = True

            if dd + 1 <= dt.year_month_day_cnt(yy, mm):
                cur = [yy, mm, dd + 1]
            elif mm + 1 <= 12:
                cur = [yy, mm + 1, 1]
            else:
                cur = [yy + 1, 1, 1]
            if (cur[0], cur[1], cur[2]) in dct and not dp[(cur[0], cur[1], cur[2])]:
                dp[(yy, mm, dd)] = True

        for _ in range(ac.read_int()):
            x, y, z = ac.read_list_ints()
            ac.st("YES" if dp.get((x, y, z), True) else "NO")

        return

    @staticmethod
    def lg_p2092(ac=FastIO()):
        # 模板：根据质数的个数来判断必胜态
        n = ac.read_int()
        lst = NumberTheory().get_prime_factor2(n)
        nums = []
        for p, c in lst:
            nums.extend([p] * c)
        if not nums or len(nums) == 1:
            ac.st(1)
            ac.st(0)
            return
        if len(nums) == 2:
            ac.st(2)
            return
        ac.st(1)
        ac.st(nums[0] * nums[1])
        return

    @staticmethod
    def lg_p2953(ac=FastIO()):
        # 模板：必胜态线性DP
        n = 1000000
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            lst = [w for w in str(i) if w != "0"]
            lst.sort()
            for w in [lst[0], lst[-1]]:
                if not dp[i - int(w)]:
                    dp[i] = 1
                    break
        for _ in range(ac.read_int()):
            if dp[ac.read_int()]:
                ac.st("YES")
            else:
                ac.st("NO")
        return


class TestGeneral(unittest.TestCase):

    def test_game_dp(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
