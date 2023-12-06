"""
Algorithm：乘法逆元、组合数求幂快速计算
Function：求逆元取模，注意取的模必须为互质数，且不能整除该质数，否则不存在对应的乘法逆元，如果不互质，需要排除公因数进行单独计算

=====================================LuoGu======================================
3811（https://www.luogu.com.cn/problem/P3811）使用乘法逆元计算
5431（https://www.luogu.com.cn/problem/P5431）使用乘法逆元计算
2613（https://www.luogu.com.cn/problem/P2613）使用乘法逆元计算
5431（https://www.luogu.com.cn/problem/P5431）转换为前缀积与后缀积计算求解

===================================CodeForces===================================
1833F（https://codeforces.com/contest/1833/problem/F）使用前缀乘积计算区间取模


"""
from collections import Counter

from src.mathmatics.multiplicative_inverse.template import MultiplicativeInverse
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p3811(ac=FastIO()):
        n, p = ac.read_list_ints()
        for i in range(1, n + 1):
            ac.st(MultiplicativeInverse().mod_reverse(i, p))
        return

    @staticmethod
    def main(ac=FastIO()):
        mod = 10 ** 9 + 7
        mi = MultiplicativeInverse()
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
                    # 乘法逆元
                    ans += pre[i + m] * mi.mod_reverse(pre[i], mod)
                    ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def lg_p5431(ac=FastIO()):
        # 模板：转换为前缀积与后缀积计算求解
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