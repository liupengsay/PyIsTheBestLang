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

from src.mathmatics.extend_gcd.template import ExtendGcd
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def ac_4299(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4299/
        tag: single_equation|ex_gcd
        """
        # 扩展欧几里得求解ax+by=n的非负整数解
        n, a, b = [ac.read_int() for _ in range(3)]
        g = math.gcd(a, b)
        if n % g:
            ac.st("NO")
        else:
            # 求解ax+by=n且x>=0和y>=0
            gcd, x1, y1 = ExtendGcd().solve_equal(a, b, n)
            low = math.ceil((-x1 * gcd) / b)
            high = (y1 * gcd) // a
            # low<=t<=high
            if low <= high:
                x = x1 + (b // gcd) * low
                ac.st("YES")
                ac.lst([x, (n - a * x) // b])
            else:
                ac.st("NO")
        return
