"""
Algorithm：ex_gcd|binary_gcd|bin_gcd|peishu_theorem
Description：single_equation

====================================LeetCode====================================
365（https://leetcode.cn/problems/water-and-jug-problem/）peishu_theorem|greedy
2543（https://leetcode.cn/contest/biweekly-contest-96/problems/check-if-point-is-reachable/）binary_gcd|ex_gcd

=====================================LuoGu======================================
P1082（https://www.luogu.com.cn/problem/P1082）same_mod|equation
P5435（https://www.luogu.com.cn/problem/P5435）binary_gcd|classical
P5582（https://www.luogu.com.cn/problem/P5582）greedy|brain_teaser|ex_gcd|peishu_theorem
P1516（https://www.luogu.com.cn/problem/P1516）single_equation


=====================================AtCoder======================================
ABC340F（https://atcoder.jp/contests/abc340/tasks/abc340_f）ex_gcd|equation|math

=====================================CodeForces======================================
1152C（https://codeforces.com/problemset/problem/1152/C）gcd_like|observation|brute_force

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
            ac.no()
        else:
            gcd, x0, y0 = lst
            low = math.ceil((-x0 * gcd) / b)
            high = (y0 * gcd) // a
            if low <= high:
                x = x0 + (b // gcd) * low
                ac.yes()
                ac.lst([x, (n - a * x) // b])
            else:
                ac.no()
        return

    @staticmethod
    def abc_340f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc340/tasks/abc340_f
        tag: ex_gcd|equation|math
        """
        x, y = ac.read_list_ints()
        if x == 0:
            if 2 % abs(y) == 0:
                ac.lst([2 // abs(y), 0])
            else:
                ac.st(-1)
            return
        if y == 0:
            if 2 % abs(x) == 0:
                ac.lst([0, 2 // abs(x)])
            else:
                ac.st(-1)
            return

        lst = GcdLike().solve_equation(y, -x, 2)
        if not lst:
            ac.st(-1)
            return
        gcd, x0, y0 = lst[:]
        ac.lst([x0, y0])
        return
