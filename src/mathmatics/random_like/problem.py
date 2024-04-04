"""
Algorithm：random_like
Description：

====================================LeetCode====================================

====================================CodeForces====================================
1914G2（https://codeforces.com/contest/1914/problem/G2）random_like|brain_teaser|range_cover

====================================AtCoder====================================
ABC272G（https://atcoder.jp/contests/abc272/tasks/abc272_g）random_guess|brute_force|num_factor|classical

=====================================LuoGu======================================



"""
import random
from collections import Counter

from src.data_structure.segment_tree.template import RangeSetRangeSumMinMax
from src.mathmatics.number_theory.template import NumFactor
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1914g2(ac=FastIO()):
        mod = 998244353
        for _ in range(ac.read_int()):

            n = ac.read_int()
            nums = ac.read_list_ints()
            rd = [random.randint(0, 10 ** 18) for _ in range(n)]
            nums = [nums[i] ^ rd[nums[i] - 1] for i in range(n * 2)]
            tree = RangeSetRangeSumMinMax(2 * n)
            start = -1
            dct = dict()
            pre = 0
            dct[pre] = -1
            ans = 1
            tot = 0
            for i, num in enumerate(nums):
                pre ^= num
                if pre in dct:
                    if dct[pre] == start:
                        cur = i - start - tree.range_sum(start + 1, i)
                        ans *= cur
                        ans %= mod
                        dct = dict()
                        pre = 0
                        dct[pre] = i
                        tot += 1
                        start = i
                    else:
                        tree.range_set(dct[pre] + 1, i, 1)
                else:
                    dct[pre] = i

            ac.lst([tot, ans])
        return

    @staticmethod
    def abc_272g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc272/tasks/abc272_g
        tag: random_guess|brute_force|num_factor|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        ceil = 10 ** 9
        nf = NumFactor()
        for _ in range(30):
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            while i == j:
                j = random.randint(0, n - 1)
            lst = nf.get_all_factor(abs(nums[i] - nums[j]))
            for m in lst:
                if 3 <= m <= ceil:
                    cur = [num % m for num in nums]
                    if max(Counter(cur).values()) * 2 > n:
                        ac.st(m)
                        return
        ac.st(-1)
        return
