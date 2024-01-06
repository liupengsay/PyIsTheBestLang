"""
Algorithm：mod_reverse|comb
Description：the reverse mod must be coprime, otherwise the gcd will deal with specially

=====================================LuoGu======================================
P3811（https://www.luogu.com.cn/problem/P3811）mod_reverse
P5431（https://www.luogu.com.cn/problem/P5431）mod_reverse
P2613（https://www.luogu.com.cn/problem/P2613）mod_reverse
P5431（https://www.luogu.com.cn/problem/P5431）prefix_suffix

===================================CodeForces===================================
1833F（https://codeforces.com/contest/1833/problem/F）prefix_mul|mod


"""
from collections import Counter

from src.mathmatics.comb_perm.template import Combinatorics
from src.mathmatics.mod_inverse.template import ModInverse
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3811_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3811
        tag: mod_reverse|classical|hard
        """
        n, p = ac.read_list_ints()
        inv = [0] * (n + 1)
        inv[1] = 1
        for i in range(2, n + 1):
            inv[i] = (p - p // i) * inv[p % i] % p
        for x in inv[1:]:
            ac.st(x)
        return

    @staticmethod
    def lg_p3811_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3811
        tag: mod_reverse|classical|hard
        """
        n, p = ac.read_list_ints()
        cb = Combinatorics(n, p)
        for x in range(1, n + 1):
            ac.st(cb.inv(x))
        return

    @staticmethod
    def main(ac=FastIO()):
        mod = 10 ** 9 + 7
        mi = ModInverse()
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            nums = ac.read_list_ints()
            cnt = Counter(nums)
            lst = sorted(list(cnt.keys()))
            ans = 0
            k = len(lst)
            # 前缀乘积
            pre = [0] * (n + 1)
            pre[0] = 1
            for i in range(k):
                pre[i + 1] = (pre[i] * cnt[lst[i]]) % mod
            for i in range(k - m + 1):
                if lst[i + m - 1] == lst[i] + m - 1:
                    # mod_reverse
                    ans += pre[i + m] * mi.mod_reverse(pre[i], mod)
                    ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def lg_p5431(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5431
        tag: prefix_suffix
        """
        # 转换为前缀积与后缀积求解
        n, p, k = ac.read_list_ints()
        a = ac.read_list_ints()
        post = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            post[i] = (post[i + 1] * a[i]) % p
        # 遍历数组
        kk = k
        pre = 1
        ans = 0
        for i in range(n):
            ans += kk * pre * post[i + 1]
            ans %= p
            kk = (kk * k) % p
            pre = (pre * a[i]) % p
        ans *= pow(pre, -1, p)
        ans %= p
        ac.st(ans)
        return
