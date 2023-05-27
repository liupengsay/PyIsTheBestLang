import math
import unittest

from functools import lru_cache

from itertools import permutations

from algorithm.src.fast_io import FastIO
from algorithm.src.mathmatics.lexico_graphical_order import LexicoGraphicalOrder

"""
算法：康托展开
功能：康托展开可以用来求一个 1~n的任意排列的排名
题目：

===================================洛谷===================================
P3014 [USACO11FEB]Cow Line S（https://www.luogu.com.cn/problem/P3014）计算全排列的排名与排名对应的全排列
P5367 【模板】康托展开（https://www.luogu.com.cn/problem/P5367）计算排列的排名

参考：OI WiKi（https://oi-wiki.org/math/combinatorics/cantor/）
"""


class CantorExpands:
    def __init__(self, n, mod=10**9 + 7):
        self.mod = mod
        self.dp = [1] * (n + 1)
        for i in range(2, n):
            self.dp[i] = i * self.dp[i - 1] % mod
        return

    def array_to_rank(self, nums):
        lens = len(nums)
        out = 0
        for i in range(lens):
            res = 0
            fact = self.dp[lens - i - 1]
            for j in range(i + 1, lens):
                if nums[j] < nums[i]:
                    res += 1
            out += res * fact
            out %= self.mod
        return out + 1

    def rank_to_array(self, n, k):
        nums = list((1, n + 1))
        k = k - 1
        out = []
        while nums:
            p = k // self.dp[n - 1]
            out.append(nums[p])
            k = k - p * self.dp[n - 1]
            nums.pop(p)
        return out


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p5367(ac=FastIO()):
        # 模板：计算数组在 1 到 n 的全排列当中的排名
        n = ac.read_int()
        nums = ac.read_list_ints()
        mod = 998244353
        ce = CantorExpands(n, mod)
        ac.st(ce.array_to_rank(nums) % mod)
        return

    @staticmethod
    def lg_p3014(ac=FastIO()):
        # 模板：康托展开也可以使用字典序贪心计算
        n, q = ac.read_ints()
        # og = LexicoGraphicalOrder()
        ct = CantorExpands(n, mod=math.factorial(n + 2))
        for _ in range(q):
            s = ac.read_str()
            lst = ac.read_list_ints()
            if s == "P":
                # ac.lst(og.get_kth_subset_perm(n, lst[0]))
                ac.lst(ct.rank_to_array(n, lst[0]))
            else:
                # ac.st(og.get_subset_perm_kth(n, lst))
                ac.st(ct.array_to_rank(lst))
        return


class TestGeneral(unittest.TestCase):

    def test_cantor_expands(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
