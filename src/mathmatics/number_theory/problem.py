"""
Algorithm：number_theory|euler_sieve|linear_sieve|prime|euler_phi|factorization|prime_factorization|base
Description：digital_dp|n-base|prime_factorization|factorization|linear_sieve|euler_phi|pollard_rho|meissel_lehmer|range_prime_count

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
P1952（https://www.luogu.com.cn/problem/P1952）n-base
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

====================================AtCoder=====================================
ABC114D（https://atcoder.jp/contests/abc114/tasks/abc114_d）prime_factorization|counter
ABC134D（https://atcoder.jp/contests/abc134/tasks/abc134_d）reverse_thinking|construction
ABC337E（https://atcoder.jp/contests/abc337/tasks/abc337_e）n-base|classical

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


"""
import math
from collections import Counter
from collections import defaultdict
from functools import reduce
from operator import mul
from sys import stdout
from typing import List

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
                ac.st("NO")
            elif len(rest) == 1:
                if rest[0][1] > 1:
                    ac.st("NO")
                else:
                    if pj.is_prime_speed(rest[0][0]):
                        ac.st("YES")
                    else:
                        ac.st("NO")
        return

    @staticmethod
    def lg_p5248(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5248
        tag: n-base|classical
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
                ac.st("yes")
            else:
                ac.st("no")
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
                            ac.st("NO")
                            return
                ac.st("YES")
                return

            check()

        return

    @staticmethod
    def ac_4484(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4487/
        tag: n-base
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
                    ac.st("YES")
                    break
                if x > a or b == 0:
                    ac.st("NO")
                    break
                if (a - x) % b == 0:
                    ac.st("YES")
                    break
                y = ac.ceil(a, b) - 1
                a -= y * b
                if y == 0:
                    ac.st("NO")
                    break

        return

    @staticmethod
    def cf_1744_e2(ac=FastIO()):
        # 因数brute_force
        for _ in range(ac.read_int()):
            a, b, c, d = ac.read_list_ints()
            lst_a = NumberTheory().get_all_factor(a)
            lst_b = NumberTheory().get_all_factor(b)

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
        # 负进制转换模板题
        lst = NumberTheory().get_k_bin_of_n(n, -2)
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
        tag: n-base|classical
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
