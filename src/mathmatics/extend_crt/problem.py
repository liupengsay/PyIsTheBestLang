"""
Algorithm：chinese_remainder_theorem|extended_chinese_remainder_theorem|ex_crt|crt
Description：equation|same_mod


====================================LeetCode====================================

=====================================LuoGu======================================
p1495（https://www.luogu.com.cn/problem/p1495）mod_coprime|chinese_reminder_theorem|classical
P4777（https://www.luogu.com.cn/problem/P4777）mod_not_coprime|crt|chinese_reminder_theorem|classical
P3868（https://www.luogu.com.cn/problem/P3868）ex_crt|chinese_reminder_theorem|classical

====================================AtCoder=====================================
ABC286F（https://atcoder.jp/contests/abc286/tasks/abc286_f）chinese_reminder_theorem|interaction|circular_section|classical
ABC371G（https://atcoder.jp/contests/abc371/tasks/abc371_g）ex_crt|implemention|greedy|classical


===================================CodeForces===================================

"""
from src.mathmatics.extend_crt.template import CRT, ExtendCRT
from src.mathmatics.number_theory.template import PrimeSieve
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc286/tasks/abc286_f
        tag: chinese_reminder_theorem|interaction|circular_section|classical
        """
        ac.flush = True
        lst = [x for x in PrimeSieve().eratosthenes_sieve(110) if x < 110]
        tot = lst[:9]
        tot[0] *= tot[0]
        tot[1] *= tot[1]
        m = sum(tot)
        assert m == 108
        nums = list(range(1, m + 1))
        pre = 0
        circle = dict()
        for num in tot:
            tmp = nums[pre:pre + num]
            nums[pre:pre + num] = tmp[1:] + tmp[:1]
            circle[pre + 1] = tmp[1:] + tmp[:1]
            pre += num
        ac.st(m)
        ac.lst(nums)
        b = ac.read_list_ints_minus_one()
        mod_res = []
        pre = 0
        for num in tot:
            tmp = b[pre:pre + num]
            mod_res.append((num, (tmp[0] - pre) % num))
            pre += num
        ans = CRT().chinese_remainder(mod_res)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1495(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1495
        tag: chinese_reminder_theorem|classical
        """
        n = ac.read_int()
        crt = CRT()
        pairs = [ac.read_list_ints() for _ in range(n)]
        ans = crt.chinese_remainder(pairs)
        ac.st(ans)
        return

    @staticmethod
    def lg_p4777(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4777
        tag: chinese_reminder_theorem|classical
        """
        n = ac.read_int()
        ex_crt = ExtendCRT()
        pairs = [ac.read_list_ints()[::-1] for _ in range(n)]
        ans = ex_crt.pipline(pairs)[0]
        ac.st(ans)
        return

    @staticmethod
    def lg_p3868(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3868
        tag: chinese_reminder_theorem|classical
        """
        ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        ex_crt = ExtendCRT()
        pairs = [[x % y, y] for x, y in zip(a, b)]
        ans = ex_crt.pipline(pairs)[0]
        ac.st(ans)
        return
