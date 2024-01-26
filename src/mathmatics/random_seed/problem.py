"""
Algorithm：random_seed
Description：

====================================LeetCode====================================

====================================CodeForces====================================
1914G2（https://codeforces.com/contest/1914/problem/G2）random_seed|brain_teaser|range_cover

=====================================LuoGu======================================

"""
import random

from src.data_structure.segment_tree.template import RangeSetRangeSumMinMax
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
                        tree.range_change(dct[pre] + 1, i, 1)
                else:
                    dct[pre] = i

            ac.lst([tot, ans])
        return
