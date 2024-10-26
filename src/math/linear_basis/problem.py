"""
Algorithm：linear_basis|kth_subset_xor|rank_of_xor
Description：subset_xor|kth_xor|rank_of_xor

=====================================LuoGu======================================
P3812（https://www.luogu.com.cn/problem/P3812）linear_basis|classical
P3857（https://www.luogu.com.cn/problem/P3857）linear_basis|classical
P4570（https://www.luogu.com.cn/problem/P4570）linear_basis|classical
P4301（https://www.luogu.com.cn/problem/P4301）linear_basis|classical
P4151（https://www.luogu.com.cn/problem/P4151）
P3265（https://www.luogu.com.cn/problem/P3265）

=====================================CodeForces======================================
CF845G（https://codeforces.com/problemset/problem/845/G）

=====================================AtCoder======================================
ABC283G（https://atcoder.jp/contests/abc283/tasks/abc283_g）linear_basis|classical
ABC236F（https://atcoder.jp/contests/abc236/tasks/abc236_f）linear_basis|mst|greed|classical

=====================================AcWing======================================
3167（https://www.acwing.com/problem/content/description/3167/）linear_basis|classical


"""

from src.math.linear_basis.template import LinearBasis, LinearBasisVector
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3812(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3812
        tag: linear_basis
        """
        ac.read_int()
        lst = ac.read_list_ints()
        linear_basis = LinearBasis(50)
        for num in lst:
            linear_basis.add(num)
        ans = linear_basis.query_max()
        ac.st(ans)
        return

    @staticmethod
    def abc_283g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc283/tasks/abc283_g
        tag: linear_basis|classical
        """
        n, ll, rr = ac.read_list_ints()
        lst = ac.read_list_ints()
        linear_basis = LinearBasis(60)
        for num in lst:
            linear_basis.add(num)
        ans = [linear_basis.query_kth_xor(k) for k in range(ll - 1, rr)]
        ac.lst(ans)
        return

    @staticmethod
    def ac_3167(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3167/
        tag: linear_basis|classical
        """
        ac.read_int()
        lst = ac.read_list_ints()
        linear_basis = LinearBasis(64)
        for num in lst:
            linear_basis.add(num)
        ans = linear_basis.query_max()
        ac.st(ans)
        return

    @staticmethod
    def lg_p3857(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3857
        tag: linear_basis|classical
        """
        n, m = ac.read_list_ints()
        linear_basis = LinearBasis(n)
        for _ in range(m):
            s = ac.read_str().replace("O", "1").replace("X", "0")
            linear_basis.add(int("0b" + s, 2))
        ans = linear_basis.tot % 2008
        ac.st(ans)
        return

    @staticmethod
    def lg_p4570(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4570
        tag: linear_basis|classical
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: -it[1])
        linear_basis = LinearBasis(64)
        ans = 0
        for num, magic in nums:
            if linear_basis.add(num):
                ans += magic
        ac.st(ans)
        return

    @staticmethod
    def lg_p4301(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4301
        tag: linear_basis|classical
        """
        ac.read_int()
        nums = ac.read_list_ints()
        nums.sort(reverse=True)
        linear_basis = LinearBasis(32)
        ans = 0
        for num in nums:
            if not linear_basis.add(num):
                ans += num
        ac.st(ans)
        return

    @staticmethod
    def lg_p3265(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3265
        tag: linear_basis|classical
        """
        n, m = ac.read_list_ints()  # TLE
        nums = [ac.read_list_ints() for _ in range(n)]
        cost = ac.read_list_ints()

        ind = list(range(n))
        ind.sort(key=lambda it: cost[it])
        linear_basis = LinearBasisVector(m)
        ans = tot = 0
        for i in ind:
            if linear_basis.add(nums[i]):
                tot += 1
                ans += cost[i]
        ac.lst([tot, ans])
        return

    @staticmethod
    def abc_236f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc236/tasks/abc236_f
        tag: linear_basis|mst|greed|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        ind = list(range(1, 1 << n))
        ind.sort(key=lambda it: nums[it - 1])
        linear = LinearBasis()
        ans = 0
        for i in ind:
            if linear.add(i):
                ans += nums[i - 1]
        ac.st(ans)
        return