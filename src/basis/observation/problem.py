"""

Algorithm：observation
Description：observation|property|data_range

====================================LeetCode====================================

=====================================LuoGu======================================

===================================CodeForces===================================
1305C（https://codeforces.com/problemset/problem/1305/C）observation|property|data_range
1705D（https://codeforces.com/problemset/problem/1705/D）observation|property
1646C（https://codeforces.com/problemset/problem/1646/C）observation|data_range
1809D（https://codeforces.com/problemset/problem/1809/D）observation|data_range|construction
1749D（https://codeforces.com/problemset/problem/1749/D）observation|data_range
1185C2（https://codeforces.com/problemset/problem/1185/C2）data_range|bucket_cnt|greedy
1616C（https://codeforces.com/problemset/problem/1616/C）compute_slope|brute_force|observation|arithmetic_sequence
1239A（https://codeforces.com/problemset/problem/1239/A）fibonacci_array|linear_dp|observation
1889B（https://codeforces.com/problemset/problem/1889/B）observation|implemention
1332B（https://codeforces.com/problemset/problem/1332/B）observation|data_range|min_prime|data_range
1583C（https://codeforces.com/problemset/problem/1583/C）observation
1605C（https://codeforces.com/problemset/problem/1605/C）observation
1634B（https://codeforces.com/problemset/problem/1634/B）observation|odd_even|data_range
1687A（https://codeforces.com/problemset/problem/1687/A）observation|greedy
1548A（https://codeforces.com/problemset/problem/1548/A）observation|brain_teaser|implemention
1408D（https://codeforces.com/problemset/problem/1408/D）brute_force|liner_dp|greedy|brain_teaser|observation

====================================AtCoder=====================================




"""
from src.math.geometry.template import Geometry
from src.math.prime_factor.template import PrimeFactor
from src.util.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1185c2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1185/C2
        tag: data_range|bucket_cnt|greedy
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        cnt = [0] * 101
        ans = []
        pre = 0
        for num in nums:
            rest = m - pre
            cur = 0
            for x in range(100, 0, -1):
                if rest >= num:
                    break
                y = min(cnt[x], (num - rest + x - 1) // x)
                cur += y
                rest += y * x
            ans.append(cur)
            cnt[num] += 1
            pre += num
        ac.lst(ans)
        return

    @staticmethod
    def cf_1616c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1616/C
        tag: compute_slope|brute_force|observation|arithmetic_sequence
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            ans = n - 1
            gm = Geometry()
            for i in range(n):
                for j in range(i + 1, n):
                    cur = gm.compute_slope(nums[i], i, nums[j], j)
                    cnt = 2
                    for k in range(j + 1, n):
                        if gm.compute_slope(nums[i], i, nums[k], k) == cur:
                            cnt += 1
                    ans = min(ans, n - cnt)
            ac.st(ans)
        return

    @staticmethod
    def cf_1239a(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1239/A
        tag: fibonacci_array|linear_dp|observation
        """
        mod = 10 ** 9 + 7
        m, n = ac.read_list_ints()
        ceil = max(m, n) + 10
        dp = [0] * ceil
        dp[0] = dp[1] = 1
        for i in range(2, ceil):
            dp[i] = (dp[i - 1] + dp[i - 2]) % mod
        ans = (dp[n] + dp[m] - 1) * 2 % mod
        ac.st(ans)
        return

    @staticmethod
    def cf_1332b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1332/B
        tag: observation|data_range|min_prime|data_range
        """
        pf = PrimeFactor(1000)
        for _ in range(ac.read_int()):
            ac.read_int()
            nums = ac.read_list_ints()
            lst = list(set([pf.min_prime[x] for x in nums]))
            dct = {num: i + 1 for i, num in enumerate(lst)}
            ans = [dct[pf.min_prime[x]] for x in nums]
            ac.st(len(dct))
            ac.lst(ans)
        return

    @staticmethod
    def main(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1408/D
        tag: brute_force|liner_dp|greedy|brain_teaser|observation
        """
        n, m = ac.read_list_ints()
        nums1 = [ac.read_list_ints() for _ in range(n)]
        nums2 = [ac.read_list_ints() for _ in range(m)]
        ceil = 10 ** 6 + 10
        right = [0] * ceil
        ans = 0
        for a, b in nums1:
            for c, d in nums2:
                if d >= b:
                    right[d - b] = max(right[d - b], c - a + 1)
                    ans = max(ans, d - b + 1)
        for i in range(ceil - 2, -1, -1):
            right[i] = max(right[i], right[i + 1])
            ans = min(ans, i + right[i])
        ac.st(ans)
        return

    @staticmethod
    def cf_1408d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1408/D
        tag: brute_force|liner_dp|greedy|brain_teaser|observation
        """
        n, m = ac.read_list_ints()
        nums1 = [ac.read_list_ints() for _ in range(n)]
        nums2 = [ac.read_list_ints() for _ in range(m)]
        ceil = 10 ** 6 + 10
        right = [0] * ceil
        ans = 0
        for a, b in nums1:
            for c, d in nums2:
                if d >= b:
                    right[d - b] = max(right[d - b], c - a + 1)
                    ans = max(ans, d - b + 1)
        for i in range(ceil - 2, -1, -1):
            right[i] = max(right[i], right[i + 1])
            ans = min(ans, i + right[i])
        ac.st(ans)
        return
