"""
Algorithm：number_theory|euler_sieve|linear_sieve|prime|euler_phi|factorization|prime_factorization|base
Description：digital_dp|n_base|prime_factorization|factorization|linear_sieve|euler_phi|pollard_rho|meissel_lehmer|range_prime_count

====================================LeetCode====================================
264（https://leetcode.cn/problems/ugly-number-ii/）pointer|dp|ugly_number|classical
1201（https://leetcode.cn/problems/ugly-number-iii/）ugly_number
313（https://leetcode.cn/problems/super-ugly-number/）ugly_number
12（https://leetcode.cn/problems/integer-to-roman/）integer_to_roman
13（https://leetcode.cn/problems/roman-to-integer/）roman_to_integer
2572（https://leetcode.cn/problems/count-the-number-of-square-free-subsets/）ag_dp|counter
1994（https://leetcode.cn/problems/the-number-of-good-subsets/）bag_dp|counter
2464（https://leetcode.cn/problems/minimum-subarrays-in-a-valid-split/）prime_factorization|counter|dp
14（https://leetcode.cn/problems/qie-fen-shu-zu/）prime_factorization|counter|dp
279（https://leetcode.cn/problems/perfect-squares/）four_square
650（https://leetcode.cn/problems/2-keys-keyboard/）prime_factorization
1390（https://leetcode.cn/problems/four-divisors/）preprocess|factorization
1819（https://leetcode.cn/problems/number-of-different-subsequences-gcds/）preprocess|factorization|brute_force|gcd
1017（https://leetcode.cn/problems/convert-to-base-2/）negative_base|classical
1073（https://leetcode.cn/problems/adding-two-negabinary-numbers/）negative_base|classical
8041（https://leetcode.cn/problems/maximum-element-sum-of-a-complete-subset-of-indices/description/）prime_factorization|hash|classical|odd

=====================================LuoGu======================================
P1865（https://www.luogu.com.cn/problem/P1865）linear_sieve|prime|binary_search|range_prime_count
P1748（https://www.luogu.com.cn/problem/P1748）heapq|implemention|pointer|inclusion_exclusion|binary_search
P2723（https://www.luogu.com.cn/problem/P2723）ugly_number
P1952（https://www.luogu.com.cn/problem/P1952）n_base
P1555（https://www.luogu.com.cn/problem/P1555）2-base|3-base
P1465（https://www.luogu.com.cn/problem/P1465）int_to_roman
P1112（https://www.luogu.com.cn/problem/P1112）brute_force
P2926（https://www.luogu.com.cn/problem/P2926）prime_factorization|counter
P5535（https://www.luogu.com.cn/problem/P5535）prime|is_prime5|greedy|brain_teaser
P1876（https://www.luogu.com.cn/problem/P1876）odd_even|factorization|classical
P1887（https://www.luogu.com.cn/problem/P1887）classical|maximum_mul
P2043（https://www.luogu.com.cn/problem/P2043）prime_factorization|prime_sieve|factorial
P2192（https://www.luogu.com.cn/problem/P2192）divide|property|classical
P7191（https://www.luogu.com.cn/problem/P7191）mod|math|factorization
P7517（https://www.luogu.com.cn/problem/P7517）prime_sieve|brute_force|factorization|counter
P7588（https://www.luogu.com.cn/problem/P7588）prime|brute_force|is_prime4
P7696（https://www.luogu.com.cn/problem/P7696）prime_factorization
P4718（https://www.luogu.com.cn/problem/P4718）pollard_rho|prime_factorization|prime
P2429（https://www.luogu.com.cn/problem/P2429）brute_force|prime_factorization|inclusion_exclusion|counter
P1069（https://www.luogu.com.cn/problem/P1069）prime_factorization|counter
P1072（https://www.luogu.com.cn/problem/P1072）brute_force|factorization
P1593（https://www.luogu.com.cn/problem/P1593）prime_factorization|fast_power|classical
P2527（https://www.luogu.com.cn/problem/P2527）ugly_number
P2557（https://www.luogu.com.cn/problem/P2557）prime_factorization|math
P4446（https://www.luogu.com.cn/problem/P4446）is_prime
P4752（https://www.luogu.com.cn/problem/P4752）is_prime
P5248（https://www.luogu.com.cn/problem/P5248）base
P5253（https://www.luogu.com.cn/problem/P5253）math
P7960（https://www.luogu.com.cn/problem/P7960）prime_sieve|preprocess
P8762（https://www.luogu.com.cn/problem/P8762）inclusion_exclusion|prefix_sum|counter
P8778（https://www.luogu.com.cn/problem/P8778）brute_force|prime_factorization|O(n^0.25)|classical
P8782（https://www.luogu.com.cn/problem/P8782）base|greedy|classical
P5091（https://www.luogu.com.cn/problem/P5091）extend_euler_theorem|classical
P1619（https://www.luogu.com.cn/problem/P1619）prime_factor|pollard_rho
P2104（https://www.luogu.com.cn/problem/P2104）stack|n_bin
P2441（https://www.luogu.com.cn/problem/P2441）implemention|data_range|observation|math
P3383（https://www.luogu.com.cn/problem/P3383）eratosthenes_sieve
P3601（https://www.luogu.com.cn/problem/P3601）euler_phi|math|number_theory
P4282（https://www.luogu.com.cn/problem/P4282）math|n_base|classical|high_precision
P1601（https://www.luogu.com.cn/problem/P1601）math|n_base|classical|high_precision
P1303（https://www.luogu.com.cn/problem/P1303）math|n_base|classical|high_precision
P6366（https://www.luogu.com.cn/problem/P6366）n_base|observation
P6539（https://www.luogu.com.cn/problem/P6539）euler_series|classical|brute_force

===================================CodeForces===================================
1771C（https://codeforces.com/problemset/problem/1771/C）pollard_rho|prime_factorization
1349A（https://codeforces.com/contest/1349/problem/A）prime_factorization|brute_force
1295D（https://codeforces.com/contest/1295/problem/D）euler_phi|n_coprime
1538D（https://codeforces.com/problemset/problem/1538/D）pollard_rho|prime_factorization
1458A（https://codeforces.com/problemset/problem/1458/A）gcd|math
1444A（https://codeforces.com/problemset/problem/1444/A）greedy|brute_force|prime_factorization
1823C（https://codeforces.com/contest/1823/problem/C）prime_factorization|greedy
1744E2（https://codeforces.com/contest/1744/problem/E2）brute_force|factorization
1612D（https://codeforces.com/contest/1612/problem/D）gcd_like
1920C（https://codeforces.com/contest/1920/problem/C）brute_force|num_factor|gcd_like
1029F（https://codeforces.com/contest/1029/problem/F）num_factor|brute_force|greedy
1154G（https://codeforces.com/contest/1154/problem/G）num_factor|brute_force|greedy|brain_teaser|classical|minimum_lcm_pair
1360H（https://codeforces.com/contest/1360/problem/H）n_base
1475G（https://codeforces.com/contest/1475/problem/G）linear_dp|brute_force|euler_series|reverse_thinking|brute_force
1512G（https://codeforces.com/contest/1512/problem/G）euler_sieve|number_theory|all_factor_sum|multiplicative_function|classical
1593D2（https://codeforces.com/contest/1593/problem/D2）brute_force|number_theory|classical
1822G2（https://codeforces.com/contest/1822/problem/G2）eratosthenes_sieve|get_all_factor_square
1811E（https://codeforces.com/contest/1811/problem/E）n_base
1878F（https://codeforces.com/contest/1878/problem/F）number_theory|brute_force
1982D（https://codeforces.com/contest/1982/problem/D）peishu_theorem|math|implemention|brute_force|prefix_sum_matrix
1656D（https://codeforces.com/problemset/problem/1656/D）math|odd_even|observation|bain_teaser
1992F（https://codeforces.com/contest/1992/problem/F）greedy|implemention|math
1361B（https://codeforces.com/problemset/problem/1361/B）observation|limited_operation|data_range|brain_teaser|math
1478D（https://codeforces.com/problemset/problem/1478/D）math|peishu_theorem
1228C（https://codeforces.com/problemset/problem/1228/C）math|num_factor|contribution_method
1601C（https://codeforces.com/contest/1061/problem/C）get_all_factor|classical
1542C（https://codeforces.com/problemset/problem/1542/C）math|inclusion_exclusion|brain_teaser|prefix_lcm
1614D2（https://codeforces.com/problemset/problem/1614/D2）euler_series|number_theory|linear_dp|classical|factor_cnt
632D（https://codeforces.com/problemset/problem/632/D）linear_dp|math|classical|euler_series
1753B（https://codeforces.com/contest/1753/problem/B）math|n_base
2013E（https://codeforces.com/contest/2013/problem/E）greedy|gcd_like|number_theory|observation
1750D（https://codeforces.com/problemset/problem/1750/D）number_theory|observation|data_range|limited_operation|inclusion_exclusion|gcd_like|num_factor|classical
687B（https://codeforces.com/problemset/problem/687/B）math|mod|lcm|classical
        
====================================AtCoder=====================================
ABC114D（https://atcoder.jp/contests/abc114/tasks/abc114_d）prime_factorization|counter
ABC134D（https://atcoder.jp/contests/abc134/tasks/abc134_d）reverse_thinking|construction
ABC337E（https://atcoder.jp/contests/abc337/tasks/abc337_e）n_base|classical
ABC304F（https://atcoder.jp/contests/abc304/tasks/abc304_f）classical|inclusion_exclusion
ABC300D（https://atcoder.jp/contests/abc300/tasks/abc300_d）brute_force|two_pointers
ABC293E（https://atcoder.jp/contests/abc293/tasks/abc293_e）power_reverse|frac_pow|classical|math|recursion|divide_conquer
ABC284D（https://atcoder.jp/contests/abc284/tasks/abc284_d）get_prime_with_pollard_rho|num_factor|classical
ABC280D（https://atcoder.jp/contests/abc280/tasks/abc280_d）prime_factorization|brain_teaser|greedy|classical
ABC259E（https://atcoder.jp/contests/abc259/tasks/abc259_e）brute_force|lcm|num_factor
ABC253D（https://atcoder.jp/contests/abc253/tasks/abc253_d）inclusion_exclusion|lcm|math|corner_case|classical
ABC250D（https://atcoder.jp/contests/abc250/tasks/abc250_d）brute_force|counter|contribution_method|math
ABC245D（https://atcoder.jp/contests/abc245/tasks/abc245_d）implemention|math|data_range|classical
ABC242F（https://atcoder.jp/contests/abc242/tasks/abc242_f）inclusion_exclusion|counter|brute_force|classical
ABC242E（https://atcoder.jp/contests/abc242/tasks/abc242_e）n_base|math
ABC233E（https://atcoder.jp/contests/abc233/tasks/abc233_e）big_number|prefix_sum|data_range
ABC230E（https://atcoder.jp/contests/abc230/tasks/abc230_e）brain_teaser|math|divide_block|template
ABC228E（https://atcoder.jp/contests/abc228/tasks/abc228_e）math|fast_power|classical
ABC210E（https://atcoder.jp/contests/abc210/tasks/abc210_e）math|brain_teaser|ring_mst
ABC356E（https://atcoder.jp/contests/abc356/tasks/abc356_e）contribution_method|math
ABC361F（https://atcoder.jp/contests/abc361/tasks/abc361_f）inclusion_exclusion|math
ABC206E（https://atcoder.jp/contests/abc206/tasks/abc206_e）inclusion_exclusion|math|contribution_method|brute_force

=====================================AcWing=====================================
99（https://www.acwing.com/problem/content/99/）a^b|math|factorization
126（https://www.acwing.com/problem/content/126/）base
198（https://www.acwing.com/problem/content/198/）counter
200（https://www.acwing.com/problem/content/200/）anti_prime_number
201（https://www.acwing.com/problem/content/description/201/）brute_force
3730（https://www.acwing.com/problem/content/description/3730/）brain_teaser|base
4002（https://www.acwing.com/problem/content/description/4002/）CF1295D
4322（https://www.acwing.com/problem/content/4322/）prime_factorization|prefix_hash|counter
4487（https://www.acwing.com/problem/content/4487/）base
4489（https://www.acwing.com/problem/content/description/4489/）prime_factorization|greedy
4625（https://www.acwing.com/problem/content/description/4625/）brain_teaser|greedy|construction
5052（https://www.acwing.com/problem/content/description/5052/）prime_factorization|comb


1（https://www.codechef.com/problems/UQR）math|brain_teaser

"""
import bisect
import math
from collections import Counter, deque
from collections import defaultdict
from functools import reduce, lru_cache
from operator import mul
from sys import stdout
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.basis.diff_array.template import PreFixSumMatrix
from src.mathmatics.comb_perm.template import Combinatorics
from src.mathmatics.gcd_like.template import GcdLike
from src.mathmatics.number_theory.template import EulerPhi, NumFactor, PrimeSieve, NumTheory, PrimeJudge, NumBase
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2572(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/count-the-number-of-square-free-subsets/
        tag: bag_dp|counter|classical|hard
        """

        dct = {2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30}
        mod = 10 ** 9 + 7
        cnt = Counter(nums)
        pre = defaultdict(int)
        for num in cnt:
            if num in dct:
                cur = pre.copy()
                for p in pre:
                    if math.gcd(p, num) == 1:
                        cur[p * num] += pre[p] * cnt[num]
                        cur[p * num] %= mod
                cur[num] += cnt[num]
                pre = cur.copy()

        p = pow(2, cnt[1], mod)
        ans = sum(pre.values()) * p
        ans += p - 1
        return ans % mod

    @staticmethod
    def cf_1295d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1295/problem/D
        tag: euler_phi|n_coprime|classical|hard|gcd|classical
        """
        for _ in range(ac.read_int()):  # gcd(a, b) = gcd(a - b, b) = gcd(a % b, b)
            a, m = ac.read_list_ints()
            g = math.gcd(a, m)
            mm = m // g
            ans = EulerPhi().euler_phi_with_prime_factor(mm)
            ac.st(ans)
        return

    @staticmethod
    def cf_1458a(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1458/A
        tag: gcd|math|classical|hard
        """
        m, n = ac.read_list_ints()  # gcd(x, y) = gcd(x - y, y)
        a = ac.read_list_ints()  # gcd(a1, a2, ... , an) = gcd(a1, a2 - a1, ... , an - a1)
        b = ac.read_list_ints()
        g = 0
        for i in range(1, m):
            g = math.gcd(g, a[i] - a[0])
        ans = [math.gcd(g, a[0] + num) for num in b]
        ac.lst(ans)
        return

    @staticmethod
    def ac_97(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/99/
        tag: math|factorization|classical|hard
        """

        mod = 9901
        a, b = ac.read_list_ints()
        if a == 1 or b == 0:
            ac.st(1)
        elif a == 0:
            ac.st(0)
        else:
            ans = 1
            gl = GcdLike()
            for p, c in NumFactor().get_prime_factor(a):
                c *= b
                if math.gcd(p - 1, mod) == 1:
                    ans *= (pow(p, c + 1, mod) - 1) * gl.mod_reverse(p - 1, mod)
                    ans %= mod
                else:
                    ans *= (c + 1)
                    ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def ac_124(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/126/
        tag: base
        """

        st = "0123456789"
        for i in range(26):
            st += chr(i + ord("A"))
        for i in range(26):
            st += chr(i + ord("a"))
        ind = {w: i for i, w in enumerate(st)}
        for _ in range(ac.read_int()):
            a, b, word = ac.read_list_strs()
            a = int(a)
            b = int(b)
            num = 0
            for w in word:
                num *= a
                num += ind[w]
            ac.lst([a, word])
            ans = ""
            while num:
                ans += st[num % b]
                num //= b
            if not ans:
                ans = "0"
            ac.lst([b, ans[::-1]])
            ac.st("")
        return

    @staticmethod
    def ac_196(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/198/
        tag: counter|classical|hard|prime_sieve
        """

        ceil = 2 ** 31 - 1
        primes = PrimeSieve().eratosthenes_sieve(int(ceil ** 0.5) + 1)
        while True:
            lst = ac.read_list_ints()
            if not lst:
                break

            left, right = lst
            dp = [0] * (right - left + 1)
            for p in primes:
                x = max(math.ceil(left / p), 2) * p
                while left <= x <= right:
                    dp[x - left] = 1
                    x += p

            rest = [x + left for x in range(right - left + 1) if not dp[x] and x + left > 1]
            if len(rest) < 2:
                ac.st("There are no adjacent primes.")
            else:
                ans1 = [rest[0], rest[1]]
                ans2 = [rest[0], rest[1]]
                m = len(rest)
                for i in range(2, m):
                    a, b = rest[i - 1], rest[i]
                    if b - a < ans1[1] - ans1[0]:
                        ans1 = [a, b]
                    if b - a > ans2[1] - ans2[0]:
                        ans2 = [a, b]
                ac.st(f"{ans1[0]},{ans1[1]} are closest, {ans2[0]},{ans2[1]} are most distant.")
        return

    @staticmethod
    def ac_198(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/200/
        tag: anti_prime_number|classical|hard|data_range|brute_force|dfs
        """

        n = ac.read_int()
        primes = PrimeSieve().eratosthenes_sieve(50)
        x = reduce(mul, primes)
        while x > n:
            x //= primes.pop()

        m = len(primes)
        ans = [1, 1]
        stack = [(1, 1, int(math.log2(n)) + 1, 0)]
        while stack:
            x, cnt, mi, i = stack.pop()
            if mi == 0 or i == m:
                if cnt > ans[1] or (cnt == ans[1] and x < ans[0]):
                    ans = [x, cnt]
                continue
            for y in range(mi, -1, -1):
                if x * primes[i] ** y <= n:
                    stack.append((x * primes[i] ** y, cnt * (y + 1), y, i + 1))
        ac.st(ans[0])
        return

    @staticmethod
    def ac_199(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/201/
        tag: brute_force|classical|hard|data_range
        """
        n, k = ac.read_list_ints()
        ans = n * k
        left = 1
        while left <= min(n, k):
            right = min(k // (k // left), n)
            ans -= (k // left) * (left + right) * (right - left + 1) // 2
            left = right + 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p1069(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1069
        tag: prime_factorization|counter|greedy
        """
        ac.read_int()
        m1, m2 = ac.read_list_ints()
        lst = NumFactor().get_prime_factor(m1)
        ans = inf
        for num in ac.read_list_ints():
            res = 0
            for p, c in lst:
                if num % p != 0:
                    break
                tmp = num
                x = 0
                while tmp % p == 0:
                    tmp //= p
                    x += 1
                res = ac.max(res, math.ceil(c * m2 / x))
            else:
                ans = ac.min(ans, res)
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p1072(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1072
        tag: brute_force|factorization
        """
        nt = NumFactor()
        for _ in range(ac.read_int()):
            a0, a1, b0, b1 = ac.read_list_ints()
            factor = [num for num in nt.get_all_with_pollard_rho(b1)
                      if num % a1 == 0 and math.gcd(num, a0) == a1
                      and b0 * num // math.gcd(num, b0) == b1]
            ac.st(len(factor))
        return

    @staticmethod
    def lg_p1593(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1593
        tag: prime_factorization|fast_power|classical|hard
        """
        mod = 9901
        a, b = ac.read_list_ints()
        if a == 1 or b == 0:
            ac.st(1)
        else:
            ans = 1
            for p, c in NumFactor().get_prime_factor(a):
                c *= b
                if math.gcd(p - 1, mod) == 1:
                    ans *= (pow(p, c + 1, mod) - 1) * pow(p - 1, -1, mod)
                    ans %= mod
                else:
                    ans *= (c + 1)
                    ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def lg_p2429(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2429
        tag: brute_force|prime_factorization|inclusion_exclusion|counter|data_range
        """

        n, m = ac.read_list_ints()
        primes = sorted(ac.read_list_ints())
        stack = [(0, 1, 0)]
        ans = 0
        mod = 376544743

        while stack:
            i, value, cnt = stack.pop()
            if value > m:
                continue
            if i == n:
                if cnt:
                    num = m // value
                    ans += value * (num * (num + 1) // 2) * (-1) ** (cnt + 1)
                    ans %= mod
                continue
            stack.append((i + 1, value * primes[i], cnt + 1))
            stack.append((i + 1, value, cnt))
        ac.st(ans)
        return

    @staticmethod
    def lg_p2527(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2527
        tag: ugly_number
        """
        _, k = ac.read_list_ints()
        primes = ac.read_list_ints()
        ans = NumTheory().nth_super_ugly_number(k + 1, primes)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2557(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2557
        tag: prime_factorization|math
        """
        a, b = ac.read_list_ints()
        cnt = dict()
        for p, c in NumFactor().get_prime_factor(a):
            cnt[p] = c
        ans = 1
        for k in cnt:
            c = cnt[k] * b
            ans *= (k ** (c + 1) - 1) // (k - 1)
        ac.st(ans)
        return

    @staticmethod
    def lg_p4446(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4446
        tag: prime_sieve|hard|classical|brute_force
        """
        ac.read_int()
        nums = ac.read_list_ints()
        prime = PrimeSieve().eratosthenes_sieve(int(max(nums) ** 0.25) + 1)
        for num in nums:
            ans = 1
            for p in prime:
                if p ** 3 > num:
                    break
                c = 0
                while num % p == 0:
                    c += 1
                    num //= p
                ans *= p ** (c // 3)

            low = 1
            high = int(num ** (1 / 3)) + 1
            while low < high - 1:
                mid = low + (high - low) // 2
                if mid ** 3 <= num:
                    low = mid
                else:
                    high = mid
            if high ** 3 == num:
                ans *= high
            elif low ** 3 == num:
                ans *= low
            ac.st(ans)
        return

    @staticmethod
    def lg_p4752(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4752
        tag: is_prime
        """
        pj = PrimeJudge()
        for _ in range(ac.read_int()):
            ac.read_list_ints()
            cnt = Counter(ac.read_list_ints())
            for num in ac.read_list_ints():
                cnt[num] -= 1
            rest = []
            for num in cnt:
                if cnt[num] and num != 1:
                    rest.append((num, cnt[num]))
            if len(rest) != 1:
                ac.no()
            elif len(rest) == 1:
                if rest[0][1] > 1:
                    ac.no()
                else:
                    if pj.is_prime_speed(rest[0][0]):
                        ac.yes()
                    else:
                        ac.no()
        return

    @staticmethod
    def lg_p5248(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5248
        tag: n_base|classical
        """
        n, fn = ac.read_list_ints()
        lst = []
        while fn:
            lst.append(fn % n)
            fn //= n
        ac.st(len(lst))
        ac.lst(lst)
        return

    @staticmethod
    def lg_p5253(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5253
        tag: math|classical|hard
        """
        n = ac.read_int()  # (x - n) * (y - n) = n ^ 2
        lst = NumFactor().get_prime_factor(n)
        ans = 1
        for _, c in lst:
            ans *= (2 * c + 1)
        ac.st(ans // 2 + 1)
        return

    @staticmethod
    def lg_p7960(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7960
        tag: prime_sieve|preprocess
        """
        n = 10 ** 7
        dp = [0] * (n + 1)
        for x in range(1, n + 1):
            if "7" in str(x):
                y = 1
                while x * y <= n:
                    dp[x * y] = 1
                    y += 1
        post = 10 ** 7 + 1
        for i in range(n, -1, -1):
            if dp[i] == 1:
                dp[i] = -1
            else:
                dp[i] = post
                post = i
        for _ in range(ac.read_int()):
            ac.st(dp[ac.read_int()])
        return

    @staticmethod
    def lg_p8778(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8778
        tag: brute_force|prime_factorization|O(n^0.25)|classical
        """

        primes = PrimeSieve().eratosthenes_sieve(4000)

        def check(xx):
            for r in range(2, 4):
                a = int(xx ** (1 / r))
                for ww in [a - 1, a, a + 1, a + 2]:
                    if ww ** r == xx:
                        return True
            return False

        n = ac.read_int()
        for _ in range(n):
            num = ac.read_int()
            flag = True
            for p in primes:
                if p > num:
                    break
                x = 0
                while num % p == 0:
                    x += 1
                    num //= p
                if x == 1:
                    flag = False
                    break
            if flag and check(num):
                ac.yes()
            else:
                ac.no()
        return

    @staticmethod
    def lc_1073(arr1: List[int], arr2: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/adding-two-negabinary-numbers/
        tag: negative_base|classical
        """

        def check(tmp):
            res = 0
            for num in tmp:
                res = (-2) * res + num
            return res

        ans = check(arr1) + check(arr2)
        return NumBase().get_k_bin_of_n(ans, -2)

    @staticmethod
    def ac_3730(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/3730/
        tag: brain_teaser|base
        """

        for _ in range(ac.read_int()):
            def check():
                n, k = ac.read_list_ints()
                cnt = Counter()
                for num in ac.read_list_ints():
                    lst = []
                    while num:
                        lst.append(num % k)
                        num //= k
                    for i, va in enumerate(lst):
                        cnt[i] += va
                        if cnt[i] > 1:
                            ac.no()
                            return
                ac.yes()
                return

            check()

        return

    @staticmethod
    def ac_4484(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4487/
        tag: n_base
        """

        for _ in range(ac.read_int()):

            def check():
                nonlocal q
                while q > 1:
                    gg = math.gcd(q, b)
                    if gg == 1:
                        break
                    q //= gg
                return q == 1

            p, q, b = ac.read_list_ints()
            g = math.gcd(p, q)
            p //= g
            q //= g

            ac.st("YES" if check() else "NO")
        return

    @staticmethod
    def ac_4486(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4489/
        tag: prime_factorization|greedy
        """

        n = ac.read_int()
        if n == 1:
            ac.lst([1, 0])
            return

        res = NumFactor().get_prime_factor(n)

        ans = 1
        x = 0
        ind = [2 ** i for i in range(32)]
        lst = []
        for p, c in res:
            ans *= p
            for i in range(32):
                if ind[i] >= c:
                    if ind[i] > c:
                        x = 1
                    lst.append(i)
                    break
        cnt = max(w for w in lst)
        if any(w < cnt for w in lst) or x:
            cnt += 1
        ac.lst([ans, cnt])
        return

    @staticmethod
    def ac_4622(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4625/
        tag: brain_teaser|greedy|construction
        """

        n = ac.read_int()
        if n < 4:
            ac.st(1)
        elif n % 2 == 0:
            ac.st(2)
        else:
            if PrimeJudge().is_prime_speed(n - 2):
                ac.st(2)
            else:
                ac.st(3)
        return

    @staticmethod
    def cf_1612d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1612/problem/D
        tag: gcd_like
        """
        for _ in range(ac.read_int()):
            a, b, x = ac.read_list_ints()
            while True:
                if a < b:
                    a, b = b, a
                if x == a or x == b:
                    ac.yes()
                    break
                if x > a or b == 0:
                    ac.no()
                    break
                if (a - x) % b == 0:
                    ac.yes()
                    break
                y = ac.ceil(a, b) - 1
                a -= y * b
                if y == 0:
                    ac.no()
                    break

        return

    @staticmethod
    def cf_1744_e2(ac=FastIO()):
        # 因数brute_force
        for _ in range(ac.read_int()):
            a, b, c, d = ac.read_list_ints()
            lst_a = NumFactor().get_all_factor(a)
            lst_b = NumFactor().get_all_factor(b)

            def check():
                for x in lst_a:
                    for y in lst_b:
                        g = x * y
                        yy = a * b // g
                        low_1 = a // g + 1
                        high_1 = c // g

                        low_2 = b // yy + 1
                        high_2 = d // yy
                        if low_2 <= high_2 and low_1 <= high_1:
                            ac.lst([low_1 * g, low_2 * yy])
                            return
                ac.lst([-1, -1])
                return

            check()
        return

    @staticmethod
    def lc_1017(n: int) -> str:
        """
        url: https://leetcode.cn/problems/convert-to-base-2/
        tag: negative_base|classical
        """
        lst = NumBase().get_k_bin_of_n(n, -2)
        return "".join(str(x) for x in lst)

    @staticmethod
    def cf_1920c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1920/problem/C
        tag: brute_force|num_factor|gcd_like
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            ans = 0
            a = ac.read_list_ints()
            seen = set()
            lst = NumFactor().get_all_factor(n)
            m = len(lst)
            for ii in range(m):
                k = lst[ii]
                if k in seen:
                    continue
                gcd = 0
                for j in range(n - k):
                    gcd = math.gcd(gcd, a[k + j] - a[j])
                    if gcd == 1:
                        break
                if gcd == 1:
                    continue
                for w in lst[ii:]:
                    if w % k == 0 and w not in seen:
                        seen.add(w)
                        ans += gcd != 1
            ac.st(ans)
        return

    @staticmethod
    def abc_337e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc337/tasks/abc337_e
        tag: n_base|classical
        """
        n = ac.read_int()
        m = n.bit_length() if n.bit_count() > 1 else n.bit_length() - 1

        dct = [[] for _ in range(m + 1)]
        for i in range(1, n + 1):
            for j in range(m):
                if i & (1 << j):
                    dct[j + 1].append(i)

        ac.st(m)
        for ls in dct[1:]:
            ac.lst([len(ls)] + ls[:])
        stdout.flush()

        s = ac.read_str()
        ans = 0
        for i in range(m):
            if s[i] == "1":
                ans |= 1 << i
        ac.st(ans if ans else n)
        stdout.flush()
        return

    @staticmethod
    def cf_1029f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1029/problem/F
        tag: num_factor|brute_force|greedy
        """
        a, b = ac.read_list_ints()
        lst_a = NumFactor().get_all_factor(a)
        lst_b = NumFactor().get_all_factor(b)
        lst_ab = NumFactor().get_all_factor(a + b)
        ma = len(lst_a)
        mb = len(lst_b)
        ans = 2 * (a + b + 1)
        i = j = 0
        pre = a + b + 1
        for x in lst_ab:
            y = (a + b) // x
            while i < ma and lst_a[i] <= x:
                pre = ac.min(pre, a // lst_a[i])
                i += 1
            while j < mb and lst_b[j] <= x:
                pre = ac.min(pre, b // lst_b[j])
                j += 1
            if pre <= y and 2 * (x + y) < ans:
                ans = 2 * (x + y)
        ac.st(ans)
        return

    @staticmethod
    def cf_1154g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1154/problem/G
        tag: num_factor|brute_force|greedy|brain_teaser|classical|minimum_lcm_pair
        """
        ac.read_int()
        nums = ac.read_list_ints()
        m = 10 ** 7 + 1
        cnt = [-2] * m
        ans = []
        res = 10 ** 14 + 1
        for num in nums:
            if cnt[num] == -1:
                if num < res:
                    res = num
                    ans = [num, num]
            else:
                cnt[num] = -1

        arr = sorted(set(nums))
        k = len(arr)
        for i, num in enumerate(arr):
            cnt[num] = i

        if k > 1:
            cur = arr[0] * arr[1] // math.gcd(arr[0], arr[1])
            if cur < res:
                ans = [arr[0], arr[1]]
                res = cur

            for i in range(2, m):
                lst = []
                for j in range(i, m, i):
                    if cnt[j] >= 0:
                        lst.append(cnt[j])
                        if len(lst) >= 2:
                            break
                if len(lst) == 2:
                    x, y = lst[0], lst[1]
                    cur = arr[x] * arr[y] // math.gcd(arr[x], arr[y])
                    if cur < res:
                        res = cur
                        ans = [arr[x], arr[y]]
                if lst:
                    if lst[0] == 0:
                        y = 1
                    else:
                        y = 0
                    x = lst[0]
                    cur = arr[x] * arr[y] // math.gcd(arr[x], arr[y])
                    if cur < res:
                        res = cur
                        ans = [arr[x], arr[y]]
        i = nums.index(ans[0])
        nums[i] = -1
        j = nums.index(ans[1])
        ac.lst(sorted([i + 1, j + 1]))
        return

    @staticmethod
    def cf_1512g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1512/problem/G
        tag: euler_sieve|number_theory|all_factor_sum|multiplicative_function|classical
        """
        ceil = 10 ** 7
        dp = [1] * (ceil + 1)
        for p in range(2, ceil + 1):
            if dp[p] == 1:
                for i in range(p, ceil + 1, p):
                    z = i // p
                    if z % p == 0:
                        dp[i] = dp[z] + (dp[z] - dp[z // p]) * p
                    else:
                        dp[i] = dp[z] * (p + 1)

        res = [-1] * (ceil + 1)
        for i in range(1, ceil + 1):
            x = dp[i]
            if x <= ceil and res[x] == -1:
                res[x] = i
        for _ in range(ac.read_int()):
            ac.st(res[ac.read_int()])
        return

    @staticmethod
    def cf_1208d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1822/problem/G2
        tag: eratosthenes_sieve|get_all_factor_square
        """
        ps = PrimeSieve()
        primes = ps.eratosthenes_sieve(1000)
        nf = NumFactor()
        for _ in range(ac.read_int()):
            ac.read_int()
            nums = ac.read_list_ints()
            nums.sort()
            cnt = dict()
            for num in nums:
                cnt[num] = cnt.get(num, 0) + 1
            ans = 0
            for num in sorted(cnt):
                if cnt[num] > 2:
                    ans += cnt[num] * (cnt[num] - 1) * (cnt[num] - 2)
                square = nf.get_all_factor_square(primes, num)
                for f in square:
                    if f > 1:
                        b = int(f ** 0.5)
                        ans += cnt[num] * cnt.get(num // b, 0) * cnt.get(num // f, 0)
            ac.st(ans)
        return

    @staticmethod
    def abc_304f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc304/tasks/abc304_f
        tag: classical|inclusion_exclusion
        """
        n = ac.read_int()
        s = ac.read_str()
        mod = 998244353
        factor = NumFactor().get_all_factor(n)
        factor.pop()
        pre = defaultdict(int)
        ans = 0
        for m in factor:
            cnt = 0
            for i in range(m):
                if any(s[j] == "." for j in range(i, n, m)):
                    continue
                cnt += 1
            cur = pow(2, cnt, mod)
            for num in pre:
                if m % num == 0:
                    cur -= pre[num]
            cur %= mod
            ans += cur
            ans %= mod
            pre[m] = cur
        ac.st(ans)
        return

    @staticmethod
    def abc_293e_1(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc293/tasks/abc293_e
        tag: power_reverse|frac_pow|classical|math|recursion|divide_conquer
        """
        a, x, m = ac.read_list_ints()
        if a == 1:
            ac.st(x % m)
            return

        def frac_mod(aa, bb, mm):
            # (aa/bb) % mod and (aa%bb=0)
            return (aa % (mm * bb)) // bb

        mod = m * (a - 1)
        ans = frac_mod((pow(a, x, mod) - 1) % mod, a - 1, m)
        ac.st(ans)
        return

    @staticmethod
    def abc_293e_2(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc293/tasks/abc293_e
        tag: power_reverse|frac_pow|classical|math|recursion|divide_conquer
        """
        a, x, m = ac.read_list_ints()

        @lru_cache(None)
        def dfs(k):
            if k == 0:
                return 1 % m
            nex = dfs(k // 2)
            res = nex * pow(a, k // 2 + 1, m) + nex
            if k % 2 == 0:
                res -= pow(a, k + 1, m)
            return res % m

        ans = dfs(x - 1)
        ac.st(ans)
        return

    @staticmethod
    def abc_284d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc284/tasks/abc284_d
        tag: get_prime_with_pollard_rho|num_factor|classical
        """
        nf = NumFactor()
        for _ in range(ac.read_int()):
            n = ac.read_int()
            ans = nf.get_prime_with_pollard_rho(n)
            p = [x for x in ans if ans[x] == 2][0]
            q = [x for x in ans if ans[x] == 1][0]
            ac.lst([p, q])
        return

    @staticmethod
    def abc_280d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc280/tasks/abc280_d
        tag: prime_factorization|brain_teaser|greedy|classical
        """
        k = ac.read_int()
        lst = NumFactor().get_prime_factor(k)
        dct = {x: y for x, y in lst}
        ans = 0
        for x in dct:
            c = dct[x]
            for y in range(x, x * k + x, x):
                i = y
                while y % x == 0:
                    c -= 1
                    y //= x
                ans = max(ans, i)
                if c <= 0:
                    break
        ac.st(ans)
        return

    @staticmethod
    def abc_253d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc253/tasks/abc253_d
        tag: inclusion_exclusion|lcm|math|corner_case|classical
        """
        n, a, b = ac.read_list_ints()
        ans = n * (n + 1) // 2
        if a != b:
            x = n // a
            ans -= x * (x + 1) * a // 2
            x = n // b
            ans -= x * (x + 1) * b // 2
            gg = math.lcm(a, b)
            x = n // gg
            ans += (x * (x + 1) * gg) // 2
            ac.st(ans)
        else:
            x = n // a
            ans -= x * (x + 1) * a // 2
            ac.st(ans)
        return

    @staticmethod
    def abc_250d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc250/tasks/abc250_d
        tag: brute_force|counter|contribution_method|math
        """
        primes = PrimeSieve().eratosthenes_sieve(10 ** 6)
        n = ac.read_int()
        ans = 0
        for q in primes:
            high = n // (q ** 3)
            high = min(high, q - 1)
            if high >= primes[0]:
                ans += bisect.bisect_right(primes, high)
        ac.st(ans)
        return

    @staticmethod
    def abc_245d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc245/tasks/abc245_d
        tag: implemention|math|data_range|classical
        """
        n, m = ac.read_list_ints()
        a = ac.read_list_ints()
        c = ac.read_list_ints()
        b = [0] * (m + 1)
        for i in range(m, -1, -1):
            pre = 0
            for ii in range(n + 1):
                jj = n + i - ii
                if i < jj <= m:
                    pre += a[ii] * b[jj]
            b[i] = (c[n + i] - pre) // a[n]
        ac.lst(b)
        return

    @staticmethod
    def abc_242f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc242/tasks/abc242_f
        tag: inclusion_exclusion|counter|brute_force|classical
        """
        m, n, b, w = ac.read_list_ints()
        mod = 998244353
        cb = Combinatorics(m * n + 10, mod)

        @lru_cache(None)
        def dfs(row, col, x):
            if x > row * col:
                return 0
            tot = cb.comb(row * col, x)
            for i in range(1, row + 1):
                for j in range(1, col + 1):
                    if i == row and j == col:
                        continue
                    tot -= cb.comb(row, i) * cb.comb(col, j) * dfs(i, j, x)
            return tot % mod

        ans = 0
        for ii in range(1, m + 1):
            for jj in range(1, n + 1):
                if (m - ii) * (n - jj) >= w:
                    ans += cb.comb(m, ii) * cb.comb(n, jj) * dfs(ii, jj, b) * cb.comb((m - ii) * (n - jj), w)
                    ans %= mod
                else:
                    break
        ac.st(ans)
        return

    @staticmethod
    def abc_242e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc242/tasks/abc242_e
        tag: n_base|math
        """
        mod = 998244353
        for _ in range(ac.read_int()):
            n = ac.read_int()
            s = ac.read_str()
            if n % 2:
                t = s[:n // 2 + 1]
            else:
                t = s[:n // 2]
            ans = 0
            for w in t:
                ans = ans * 26 + ord(w) - ord("A")
                ans %= mod
            ans += 1
            if n % 2 and t + t[:-1][::-1] > s:
                ans -= 1
            elif n % 2 == 0 and t + t[::-1] > s:
                ans -= 1
            ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def abc_233e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc233/tasks/abc233_e
        tag: big_number|prefix_sum|data_range
        """
        s = [int(w) for w in ac.read_str()]
        pre = ac.accumulate(s)
        ans = []
        n = len(s)
        x = 0
        for i in range(n - 1, -1, -1):
            x += pre[i + 1]
            ans.append(x % 10)
            x //= 10
        while x:
            ans.append(x % 10)
            x //= 10
        ans.reverse()
        ac.st("".join(str(x) for x in ans))
        return

    @staticmethod
    def abc_230e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc230/tasks/abc230_e
        tag: brain_teaser|math|divide_block|template
        """

        n = ac.read_int()
        ans = 0
        ll = 1
        while ll <= n:
            rr = n // (n // ll)
            ans += (rr - ll + 1) * (n // ll)
            ll = rr + 1
        ac.st(ans)
        return

    @staticmethod
    def abc_228e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc228/tasks/abc228_e
        tag: math|fast_power|classical
        """
        n, k, m = ac.read_list_ints()
        mod = 998244353
        if m % mod == 0:
            ac.st(0)
        else:
            ans = pow(m, pow(k, n, mod - 1), mod)
            ac.st(ans)
        return

    @staticmethod
    def abc_210e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc210/tasks/abc210_e
        tag: math|brain_teaser|ring_mst
        """
        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(m)]
        nums.sort(key=lambda it: it[1])
        ans = 0
        for a, c in nums:
            g = math.gcd(a, n)
            ans += c * (n - g)
            n = g
        ac.st(ans if n == 1 else -1)
        return

    @staticmethod
    def abc_356e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc356/tasks/abc356_e
        tag: contribution_method|math
        """
        ac.read_int()
        nums = ac.read_list_ints()
        ceil = 10 ** 6 + 1
        cnt = [0] * (ceil + 1)
        for num in nums:
            cnt[num] += 1
        ans = 0
        pre = ac.accumulate(cnt)
        for x in range(1, ceil + 1):
            if cnt[x]:
                for y in range(x, ceil + 1, x):
                    low = y
                    high = min(y + x - 1, ceil)
                    ans += (low // x) * cnt[x] * (pre[high + 1] - pre[low])
                ans -= cnt[x] * cnt[x]
                if cnt[x] > 1:
                    ans += cnt[x] * (cnt[x] - 1) // 2
        ac.st(ans)
        return

    @staticmethod
    def cc_1(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/UQR
        tag: math|brain_teaser
        """
        for _ in range(ac.read_int()):
            n, a, b = ac.read_list_ints()
            if a == b:
                ac.st(n // a)
            else:
                a = min(a, b)
                k = n // a
                if k == 0:
                    ac.st(0)
                else:
                    if k <= a:
                        ans = k * (k + 1) // 2 - 1
                    else:
                        ans = a * (a + 1) // 2 - 1 + (k - a) * a
                    ans += min(n % a, k) + 1
                    ac.st(ans)
        return

    @staticmethod
    def cf_1982d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1982/problem/D
        tag: peishu_theorem|math|implemention|brute_force|prefix_sum_matrix
        """
        for _ in range(ac.read_int()):
            m, n, k = ac.read_list_ints()
            grid = [ac.read_list_ints() for _ in range(m)]
            st = [[2 * int(w) - 1 for w in ac.read_str()] for _ in range(m)]
            ans = g = 0
            for i in range(m):
                for j in range(n):
                    ans += grid[i][j] * st[i][j]
            pre = PreFixSumMatrix(st)
            for i in range(m - k + 1):
                for j in range(n - k + 1):
                    g = math.gcd(g, -pre.query(i, j, i + k - 1, j + k - 1))
            ac.st("YES" if g == ans == 0 or (g and ans % g == 0) else "NO")
        return

    @staticmethod
    def cf_1656d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1656/D
        tag: math|odd_even|observation|bain_teaser
        """
        for _ in range(ac.read_int()):
            b = ac.read_int() * 2
            a = 1
            while b % 2 == 0:
                b //= 2
                a *= 2
            if a < b:
                ac.st(a)
            elif a > b > 1:
                ac.st(b)
            else:
                ac.st(-1)
        return

    @staticmethod
    def cf_1992f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1992/problem/F
        tag: greedy|implemention|math
        """

        for _ in range(ac.read_int()):
            n, x = ac.read_list_ints()
            nums = ac.read_list_ints()
            ans = cnt = 0
            pre = {1}
            for num in nums:
                if any(p * num == x for p in pre) or x == num:
                    ans += cnt
                    if x != num:
                        cnt = 1
                        pre = {num} if x % num == 0 else set()
                    else:
                        cnt = 0
                        pre = {1}
                else:
                    pre = {(p * num) for p in pre if x % (p * num) == 0} | pre
                    if x % num == 0:
                        pre.add(num)
                    cnt = 1
            ac.st(ans + cnt)
        return

    @staticmethod
    def cf_1361b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1361/B
        tag: observation|limited_operation|data_range|brain_teaser|math
        """
        mod = 1000000007
        for _ in range(ac.read_int()):
            n, p = ac.read_list_ints()
            nums = ac.read_list_ints()
            if p == 1:
                ac.st(n % 2)
            else:
                ceil = 25
                dct = dict()
                for num in nums:
                    dct[num] = dct.get(num, 0) + 1
                lst = sorted(list(dct.keys()), reverse=True)
                m = len(lst)
                for i, x in enumerate(lst):
                    if dct[x] % 2 == 0:
                        continue
                    pre = 0
                    target = 1
                    for y in range(1, ceil + 1):
                        pre *= p
                        target *= p
                        if pre + dct.get(x - y, 0) >= target:
                            need = target - pre
                            dct[x - y] -= need
                            for w in range(1, y):
                                if x - w in dct:
                                    dct[x - w] = 0
                            break
                        pre += dct.get(x - y, 0)
                    else:
                        ans = pow(p, x, mod)
                        for j in range(i + 1, m):
                            ans -= dct[lst[j]] * pow(p, lst[j], mod)
                            ans %= mod
                        ans %= mod
                        ac.st(ans)
                        break
                else:
                    ac.st(0)
        return

    @staticmethod
    def cf_1478d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1478/D
        tag: math|peishu_theorem
        """
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            nums = ac.read_list_ints()
            g = 0
            for i in range(1, n):
                g = math.gcd(nums[i] - nums[0], g)
            ac.st("YES" if (k - nums[0]) % g == 0 else "NO")
        return

    @staticmethod
    def lg_p5091(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5091
        tag: extend_euler_theorem|classical
        """
        a, m, s = ac.read_list_strs()
        a = int(a)
        m = int(m)
        phi = EulerPhi().euler_phi_with_prime_factor(m)
        b = flag = 0
        for w in s:
            b *= 10
            b += int(w)
            if b >= phi:
                flag = 1
                b %= phi
        if flag:
            b += phi
        ans = pow(a, b, m)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1619(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1619
        tag: prime_factor|pollard_rho
        """
        while True:
            s = ac.read_str()
            s = "".join(w for w in s if w.isdigit())
            ac.st("Enter the number=")
            if not s:
                break
            lst = deque(s)
            while len(lst) >= 2 and lst[0] == "0":
                lst.popleft()
            s = "".join(lst)
            if len(s) >= 10 or int(s) > 4 * 10 ** 7:
                ac.st("Prime? No!")
                ac.st("The number is too large!")
            else:
                s = int(s)
                if s <= 1:
                    ac.st("Prime? No!")
                else:
                    if s <= 10 ** 6:
                        lst = NumFactor().get_prime_factor(s)
                    else:
                        dct = NumFactor().get_prime_with_pollard_rho(s)
                        lst = [(p, dct[p]) for p in sorted(dct)]
                    if len(lst) == 1 and lst[0][1] == 1:
                        ac.st("Prime? Yes!")
                    else:
                        ac.st("Prime? No!")
                        lst = [f"{p}^{c}" for p, c in lst]
                        ac.st(f"{s}=" + "*".join(lst))
            ac.st("")
        return

    @staticmethod
    def lg_p2441(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2441
        tag: implemention|data_range|observation|math
        """
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        parent = [-1] * n
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            parent[y] = x
        for _ in range(k):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x = lst[1] - 1
                num = nums[x]
                ans = -1
                while parent[x] != -1:
                    x = parent[x]
                    if math.gcd(nums[x], num) > 1:
                        ans = x + 1
                        break
                ac.st(ans)
            else:
                nums[lst[1] - 1] = lst[2]
        return

    @staticmethod
    def lg_p3601(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3601
        tag: euler_phi|math|number_theory
        """
        low, high = ac.read_list_ints()

        n = int(math.sqrt(high)) + 1
        primes = PrimeSieve().eratosthenes_sieve(n + 1)

        euler_phi = list(range(low, high + 1))
        rest = list(range(low, high + 1))

        for p in primes:
            for a in range(low // p, high // p + 2):
                if low <= a * p <= high:
                    num = a * p
                    euler_phi[num - low] *= (p - 1) / p
                    while rest[num - low] % p == 0:
                        rest[num - low] //= p
        for num in range(low, high + 1):
            if rest[num - low] > 1:
                euler_phi[num - low] *= (rest[num - low] - 1) / rest[num - low]

        mod = 666623333
        ans = (high - low + 1) * (low + high) // 2
        ans %= mod
        for num in euler_phi:
            ans -= int(num)
            ans %= mod
        ac.st(ans)
        del euler_phi
        return

    @staticmethod
    def lg_p4282(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4282
        tag: math|n_base|classical
        """
        n = ac.read_int()
        t = [0] + ac.read_list_ints()
        a = [0] + ac.read_list_ints()
        op = ac.read_str()
        b = [0] + ac.read_list_ints()
        ans = [0] * (n + 1)
        if op == "+":
            for i in range(n, 0, -1):
                ans[i] += a[i] + b[i]
                ans[i - 1] += ans[i] // t[i]
                ans[i] %= t[i]
        else:
            for i in range(n, 0, -1):
                ans[i] += a[i] - b[i]
                if ans[i] < 0:
                    ans[i - 1] -= 1
                    ans[i] += t[i]
        ac.lst(ans[1:])
        return

    @staticmethod
    def cf_1601c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1061/problem/C
        tag: get_all_factor|classical
        """
        ac.read_int()
        dp = [0] * (10 ** 6 + 1)
        dp[0] = 1
        mod = 10 ** 9 + 7
        nums = ac.read_list_ints()
        nm = NumFactor()
        for num in nums:
            factor = nm.get_all_factor(num)
            for f in factor[::-1]:
                dp[f] += dp[f - 1]
                dp[f] %= mod
        ans = sum(dp) - 1
        ac.st(ans % mod)
        return

    @staticmethod
    def lg_p6366(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6366
        tag: n_base|observation
        """
        s = ac.read_str()
        nums = []
        for w in s:
            val = int(w, 16)
            for j in range(3, -1, -1):
                if (val >> j) & 1:
                    nums.append(1)
                elif nums:
                    nums.append(0)
        if not nums:
            nums = [0]
        ans = inf
        cur = 0
        lst = nums[:]
        n = len(lst)
        if n == 1:
            ac.st(nums[0])
            return
        for i in range(1, n):
            if lst[i - 1]:
                cur += 1
                lst[i - 1] = 1 - lst[i - 1]
                lst[i] = 1 - lst[i]
                if i + 1 < n:
                    lst[i + 1] = 1 - lst[i + 1]
        if sum(lst) == 0:
            ans = cur

        cur = 1
        lst = nums[:]
        lst[0] = 1 - lst[0]
        lst[1] = 1 - lst[1]
        n = len(lst)
        for i in range(1, n):
            if lst[i - 1]:
                cur += 1
                lst[i - 1] = 1 - lst[i - 1]
                lst[i] = 1 - lst[i]
                if i + 1 < n:
                    lst[i + 1] = 1 - lst[i + 1]
        if sum(lst) == 0:
            ans = min(ans, cur)
        ac.st(ans if ans < inf else "No")
        return

    @staticmethod
    def lg_p6539(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6539
        tag: euler_series|classical|brute_force
        """
        ac.read_int()
        nums = ac.read_list_ints()
        ceil = 2 * 10 ** 6
        cnt = [0] * (ceil + 1)
        for num in nums:
            cnt[num] += 1
        ans = 0
        for x in range(1, ceil + 1):
            cur = 0
            for y in range(x, ceil + 1, x):
                cur += cnt[y]
            if cur >= 2:
                ans = max(ans, cur * x)
        ac.st(ans)
        return

    @staticmethod
    def abc_361f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc361/tasks/abc361_f
        tag: inclusion_exclusion|math
        """
        n = ac.read_int()
        m = 65
        cnt = [0] * m

        def check(x):
            return x ** b <= n

        for b in range(2, m):
            ans = BinarySearch().find_int_right(1, n, check)
            if ans >= 2:
                cnt[b] = ans - 1
        for i in range(m - 1, 1, -1):
            for j in range(i * 2, m, i):
                cnt[i] -= cnt[j]
        ac.st(sum(cnt) + 1)
        return

    @staticmethod
    def abc_206e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc206/tasks/abc206_e
        tag: inclusion_exclusion|math|contribution_method|brute_force
        """
        ll, rr = ac.read_list_ints()
        cnt = [0] * (rr + 1)
        rest = 0
        for num in range(rr, 1, -1):
            low = ll + (num - ll % num) if ll % num else ll
            high = rr
            if low <= high:
                c = high // num - low // num + 1
                cnt[num] = c * (c - 1) // 2
                rest += c - 1 if low <= num <= high else 0
                for x in range(num + num, rr + 1, num):
                    cnt[num] -= cnt[x]
        ans = sum(cnt) * 2 - rest * 2
        ac.st(ans)
        return

    @staticmethod
    def cf_1542c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1542/C
        tag: math|inclusion_exclusion|brain_teaser|prefix_lcm
        """

        pre = [1] * 42
        for i in range(2, 42):
            pre[i] = math.lcm(pre[i - 1], i)
        assert pre[-1] >= 10 ** 16
        mod = 10 ** 9 + 7
        for _ in range(ac.read_int()):
            n = ac.read_int()
            ans = 0
            for x in range(1, 41):
                ans += (x + 1) * (n // pre[x] - n // pre[x + 1])
            ac.st(ans % mod)
        return

    @staticmethod
    def cf_1614d2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1614/D2
        tag: euler_series|number_theory|linear_dp|classical|factor_cnt
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        ceil = 2 * 10 ** 7
        cnt = [0] * (ceil + 1)
        for num in nums:
            cnt[num] += 1
        factor = [0] * (ceil + 1)
        factor[1] = n
        for x in range(2, ceil + 1):
            for y in range(x, ceil + 1, x):
                factor[x] += cnt[y]

        dp = [0] * (ceil + 1)
        for x in range(ceil, 0, -1):
            dp[x] = x * factor[x]
            for y in range(x * 2, ceil + 1, x):
                dp[x] = max(dp[x], dp[y] + (factor[x] - factor[y]) * x)
        ac.st(dp[1])
        return

    @staticmethod
    def cf_2013e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/2013/problem/E
        tag: greedy|gcd_like|number_theory|observation
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            arr = sorted(list(set(nums)))
            g = reduce(math.gcd, arr)
            m = len(arr)
            use = [0] * m
            use[0] = 1
            pre = arr[0]
            ans = pre
            while True:
                cur = inf
                nex = -1
                for i in range(m):
                    if not use[i] and math.gcd(pre, arr[i]) < cur:
                        cur = math.gcd(pre, arr[i])
                        nex = i
                if nex == -1:
                    break
                use[nex] = 1
                pre = cur
                ans += pre
                if pre == g:
                    break
            ac.st(ans + (n - sum(use)) * g)
        return

    @staticmethod
    def cf_632d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/632/D
        tag: linear_dp|math|classical|euler_series
        """
        n, m = ac.read_list_ints()
        cnt = [0] * (m + 1)
        nums = ac.read_list_ints()
        for num in nums:
            if num <= m:
                cnt[num] += 1
        dp = [0] * (m + 1)
        for x in range(1, m + 1):
            if cnt[x]:
                for y in range(x, m + 1, x):
                    dp[y] += cnt[x]
        ceil = dp.index(max(dp))
        lst = []
        pre = 1
        for i, num in enumerate(nums):
            if num <= m and ceil % num == 0:
                lst.append(i + 1)
                pre = math.lcm(pre, num)
        ac.lst([pre, dp[ceil]])
        ac.lst(lst)
        return

    @staticmethod
    def cf_1750d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1750/D
        tag: number_theory|observation|data_range|limited_operation|inclusion_exclusion|gcd_like|num_factor|classical
        """
        mod = 998244353
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            nums = ac.read_list_ints()
            ans = 1
            for i in range(1, n):
                if nums[i - 1] % nums[i]:
                    ac.st(0)
                    break
                high = m // nums[i]
                low = 1
                if low > high:
                    ac.st(0)
                    break
                p = nums[i - 1] // nums[i]
                lst = NumFactor.get_all_factor(p)[1:]
                k = len(lst)
                cnt = [0] * k
                for j in range(k - 1, -1, -1):
                    cnt[j] = high // lst[j]
                    for r in range(j + 1, k):
                        if lst[r] % lst[j] == 0:
                            cnt[j] -= cnt[r]
                pre = high - low + 1 - sum(cnt)
                ans *= pre
                ans %= mod
            else:
                ac.st(ans)
        return

    @staticmethod
    def cf_687b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/687/B
        tag: math|mod|lcm|classical
        """
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        pre = 1
        for num in nums:
            pre = math.lcm(pre, num)
            pre %= k
        ac.st("Yes" if pre % k == 0 else "No")
        return
    