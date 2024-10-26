"""
Algorithm：random_like
Description：

====================================LeetCode====================================

====================================CodeForces====================================
1914G2（https://codeforces.com/contest/1914/problem/G2）random_like|brain_teaser|range_cover

====================================AtCoder====================================
ABC272G（https://atcoder.jp/contests/abc272/tasks/abc272_g）random_guess|brute_force|num_factor|classical
ABC238G（https://atcoder.jp/contests/abc238/tasks/abc238_g）random_hash|prime_hash|classical|sqrt_decomposition|offline_query|classical

=====================================LuoGu======================================



"""
import random
from collections import Counter

from src.math.number_theory.template import NumFactor
from src.math.prime_factor.template import PrimeFactor
from src.structure.segment_tree.template import RangeSetRangeSumMinMax
from src.util.fast_io import FastIO


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

    @staticmethod
    def abc_238g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc238/tasks/abc238_g
        tag: random_hash|prime_hash|classical|sqrt_decomposition|offline_query|classical
        """
        ceil = 10 ** 6
        pf = PrimeFactor(ceil)
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        prime_hash = [(0, 0, 0) for _ in range(ceil + 1)]
        for i in range(ceil + 1):
            if pf.min_prime[i] == i > 1:
                h1 = random.getrandbits(64)
                h2 = random.getrandbits(64)
                h3 = h1 ^ h2
                prime_hash[i] = (h1, h2, h3)

        pre_xor = [0] * (n + 1)
        prime_cnt = [0] * (ceil + 1)
        for i, num in enumerate(nums):
            cur = 0
            while num > 1:
                p = pf.min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                for _ in range(cnt % 3):
                    cur ^= prime_hash[p][prime_cnt[p] % 3]
                    prime_cnt[p] += 1
            pre_xor[i + 1] = pre_xor[i] ^ cur
        for _ in range(q):
            ll, rr = ac.read_list_ints_minus_one()
            if pre_xor[ll] == pre_xor[rr + 1]:
                ac.yes()
            else:
                ac.no()
        return
