"""
Algorithm：ex_gcd|binary_gcd|bin_gcd|peishu_theorem
Description：single_equation

====================================LeetCode====================================
365（https://leetcode.cn/problems/water-and-jug-problem/）peishu_theorem|greedy
2543（https://leetcode.cn/contest/biweekly-contest-96/problems/check-if-point-is-reachable/）binary_gcd|ex_gcd

=====================================LuoGu======================================
P1082（https://www.luogu.com.cn/problem/P1082）same_mod|equation
P5435（https://www.luogu.com.cn/problem/P5435）binary_gcd
P5582（https://www.luogu.com.cn/problem/P5582）greedy|brain_teaser|ex_gcd
P1516（https://www.luogu.com.cn/problem/P1516）single_equation


=====================================AcWing=====================================
4299（https://www.acwing.com/problem/content/4299/）single_equation|ex_gcd

"""
import math

from src.mathmatics.gcd_like.template import GcdLike
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def ac_4299(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4299/
        tag: single_equation|extend_gcd
        """
        n, a, b = [ac.read_int() for _ in range(3)]
        lst = GcdLike().solve_equation(a, b, n)
        if not lst:
            ac.st("NO")
        else:
            gcd, x0, y0 = lst
            low = math.ceil((-x0 * gcd) / b)
            high = (y0 * gcd) // a
            if low <= high:
                x = x0 + (b // gcd) * low
                ac.st("YES")
                ac.lst([x, (n - a * x) // b])
            else:
                ac.st("NO")
        return
