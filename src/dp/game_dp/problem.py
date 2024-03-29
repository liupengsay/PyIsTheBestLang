"""
Algorithm：game_dp|wining_state|lose
Description：brute_force|interval_dp|implemention|greedy

====================================LeetCode====================================
375（https://leetcode.cn/problems/guess-number-higher-or-lower-ii/）interval_dp|classical|game_dp
1140（https://leetcode.cn/problems/stone-game-ii/）prefix_sum|linear_dp

=====================================LuoGu======================================
P1290（https://www.luogu.com.cn/problem/P1290）classical|game_dp
P5635（https://www.luogu.com.cn/problem/P5635）game_dp|implemention
P3150（https://www.luogu.com.cn/problem/P3150）game_dp|implemention|odd_even
P4702（https://www.luogu.com.cn/problem/P4702）game_dp|implemention|odd_even
P1247（https://www.luogu.com.cn/problem/P1247）nim|game_dp|xor
P1512（https://www.luogu.com.cn/problem/P1512）game_dp|date
P2092（https://www.luogu.com.cn/problem/P2092）prime|game_dp
P2953（https://www.luogu.com.cn/problem/P2953）game_dp|winning_state|liner_dp

=====================================AcWing=====================================
4005（https://www.acwing.com/problem/content/description/4008/）classical|game_dp|brain_teaser|classification_discussion

"""
from functools import lru_cache
from functools import reduce
from operator import xor

from src.dp.game_dp.template import DateTime
from src.mathmatics.number_theory.template import NumFactor
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1290(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1290
        tag: classical|game_dp
        """
        n = ac.read_int()
        for _ in range(n):
            x, y = ac.read_list_ints()

            @lru_cache(None)
            def dfs(a, b):
                if a < b:
                    a, b = b, a
                if a % b == 0:
                    return True
                if a // b >= 2:
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
        """
        url: https://www.luogu.com.cn/problem/P1247
        tag: nim|game_dp|xor
        """
        k = ac.read_int()
        nums = ac.read_list_ints()
        x = reduce(xor, nums)
        if x == 0:
            ac.st("lose")
        else:
            for i in range(k):
                if nums[i] ^ x < nums[i]:
                    res = [nums[i] - (nums[i] ^ x), i + 1]
                    ac.lst(res)
                    nums[i] = x ^ nums[i]
                    ac.lst(nums)
                    return
        return

    @staticmethod
    def lg_p1512(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1512
        tag: game_dp|date
        """
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
        """
        url: https://www.luogu.com.cn/problem/P2092
        tag: prime|game_dp
        """
        n = ac.read_int()
        lst = NumFactor().get_prime_factor(n)
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
        """
        url: https://www.luogu.com.cn/problem/P2953
        tag: game_dp|winning_state|liner_dp
        """
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
