"""
Algorithm：peishu_theorem
Description：ax+by=gcd(a,b)  a!=0 or b!=0

====================================LeetCode====================================
1250（https://leetcode.cn/problems/check-if-it-is-a-good-array/）gcd|peishu_theorem|classical

===================================CodeForces===================================
1478D（https://codeforces.com/contest/1478/problem/D）peishu_theorem|number_theory|math
510D（https://codeforces.com/problemset/problem/510/D）peishu_theorem|linear_dp|observation

=====================================LuoGu======================================
P4549（https://www.luogu.com.cn/problem/P4549）gcd|peishu_theorem
P8646（https://www.luogu.com.cn/problem/P8646）peishu_theorem|bag_dp

"""
import math
from collections import defaultdict

from typing import List

from src.math.peishu_theorem.template import PeiShuTheorem
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1250(nums: List[int]) -> bool:
        """
        url: https://leetcode.cn/problems/check-if-it-is-a-good-array/
        tag: gcd|peishu_theorem|classical
        """
        # 转化为peishu_theorem|数组最大公约数是否等于 1 求解
        return PeiShuTheorem().get_lst_gcd(nums) == 1

    @staticmethod
    def lg_p4549(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4549
        tag: gcd|peishu_theorem
        """
        # 转化为peishu_theorem|数组最大公约数求解
        ac.read_int()
        nums = ac.read_list_ints()
        ac.st(PeiShuTheorem().get_lst_gcd(nums))
        return

    @staticmethod
    def lg_p8646(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8646
        tag: peishu_theorem|bag_dp
        """
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        nums.sort()
        s = 10 ** 4
        dp = [0] * (s + 1)
        dp[0] = 1
        for i in range(1, s + 1):
            for num in nums:
                if num > i:
                    break
                if dp[i - num]:
                    dp[i] = 1
        ans = s + 1 - sum(dp)
        if PeiShuTheorem().get_lst_gcd(nums) != 1:
            ac.st("INF")
        else:
            ac.st(ans)
        return

    @staticmethod
    def cf_510d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/510/D
        tag: peishu_theorem|linear_dp|observation
        """
        n = ac.read_int()
        ll = ac.read_list_ints()
        cc = ac.read_list_ints()
        pre = defaultdict(lambda: math.inf)
        for i in range(n):
            cur = pre.copy()
            for p in pre:
                g = math.gcd(p, ll[i])
                cur[g] = min(cur[g], pre[p] + cc[i])
            cur[ll[i]] = min(cur[ll[i]], cc[i])
            pre = cur
        ac.st(pre[1] if pre[1] < math.inf else -1)
        return
