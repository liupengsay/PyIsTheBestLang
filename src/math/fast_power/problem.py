"""
Algorithm：fast_power|matrix_fast_power|dp|mod_reverse
Description：mod|power|doubling|multiplication_method

====================================LeetCode====================================
450（https://leetcode.cn/problems/number-of-distinct-binary-strings-after-applying-operations/）brain_teaser|fast_power
1931（https://leetcode.cn/problems/painting-a-grid-with-three-different-colors/）matrix_fast_power|dp
2851（https://leetcode.cn/problems/string-transformation/description/）kmp|matrix_fast_power|classical
1622（https://leetcode.cn/problems/fancy-sequence/description/）reverse_thinking|mod_reverse|inclusion_exclusion
552（https://leetcode.cn/problems/student-attendance-record-ii）matrix_fast_power|dp
100472（https://leetcode.cn/problems/total-characters-in-string-after-transformations-ii/）matrix_fast_power|classical

=====================================LuoGu======================================
P1630（https://www.luogu.com.cn/problem/P1630）fast_power|counter|mod
P1939（https://www.luogu.com.cn/problem/P1939）matrix_fast_power
P1962（https://www.luogu.com.cn/problem/P1962）matrix_fast_power
P3390（https://www.luogu.com.cn/problem/P3390）matrix_fast_power
P3811（https://www.luogu.com.cn/problem/P3811）mod_reverse
P5775（https://www.luogu.com.cn/problem/P5775）implemention|prefix_sum|matrix_fast_power|implemention
P6045（https://www.luogu.com.cn/problem/P6045）brain_teaser|counter|fast_power|brute_force
P6075（https://www.luogu.com.cn/problem/P6075）comb|counter|fast_power
P6392（https://www.luogu.com.cn/problem/P6392）math|fast_power
P1045（https://www.luogu.com.cn/problem/P1045）math|fast_power
P3509（https://www.luogu.com.cn/problem/P3509）two_pointers|implemention|fast_power
P1349（https://www.luogu.com.cn/problem/P1349）matrix_fast_power
P2233（https://www.luogu.com.cn/problem/P2233）matrix_fast_power
P2613（https://www.luogu.com.cn/problem/P2613）mod_reverse
P3758（https://www.luogu.com.cn/problem/P3758）matrix_dp|matrix_fast_power
P5789（https://www.luogu.com.cn/problem/P5789）matrix_dp|matrix_fast_power
P5343（https://www.luogu.com.cn/problem/P5343）linear_dp|matrix_fast_power
P8557（https://www.luogu.com.cn/problem/P8557）brain_teaser|fast_power|counter
P8624（https://www.luogu.com.cn/problem/P8624）matrix_dp|matrix_fast_power
P1306（https://www.luogu.com.cn/problem/P1306）matrix_fast_power|math|fibonacci_property|classical
P2174（https://www.luogu.com.cn/problem/P2174）mod_reverse


=====================================CodeForces=====================================
1117D（https://codeforces.com/problemset/problem/1117/D）matrix_fast_power|classical|observation


=====================================AcWing=====================================
27（https://www.acwing.com/problem/content/26/）float_fast_power|classical

=====================================AtCoder=====================================
ABC236G（https://atcoder.jp/contests/abc236/tasks/abc236_g）matrix_fast_power|matrix_fast_power_min|brain_teaser|classical
ABC204F（https://atcoder.jp/contests/abc204/tasks/abc204_f）matrix_fast_power|bag_dp|brute_force|build_graph
ABC199F（https://atcoder.jp/contests/abc199/tasks/abc199_f）expectation|matrix_fast_power|classical

"""
import math
from typing import List

from src.math.fast_power.template import MatrixFastPower, FastPower, MatrixFastPowerFlatten, MatrixFastPowerMin
from src.string.kmp.template import KMP
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2851(s: str, t: str, k: int) -> int:
        """
        url: https://leetcode.cn/problems/string-transformation/description/
        tag: kmp|matrix_fast_power|classical
        """
        mod = 10 ** 9 + 7
        n = len(s)
        kmp = KMP()
        z = kmp.prefix_function(t + "#" + s + s)
        p = sum(z[i] == n for i in range(2 * n, 3 * n))
        q = n - p
        mat = [[p - 1, p], [q, q - 1]]
        vec = [1, 0] if z[2 * n] == n else [0, 1]
        res = MatrixFastPower().matrix_pow(mat, k, mod)
        ans = vec[0] * res[0][0] + vec[1] * res[0][1]
        return ans % mod

    @staticmethod
    def lg_p1045(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1045
        tag: math|fast_power
        """

        p = ac.read_int()
        ans1 = int(p * math.log10(2)) + 1
        ans2 = pow(2, p, 10 ** 501) - 1
        ans2 = str(ans2)[-500:]
        ac.st(ans1)
        ans2 = "0" * (500 - len(ans2)) + ans2
        for i in range(0, 500, 50):
            ac.st(ans2[i:i + 50])
        return

    @staticmethod
    def lg_p1630(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1630
        tag: fast_power|counter|mod|brain_teaser|permutation_circle|classical|hard
        """

        mod = 10 ** 4
        for _ in range(ac.read_int()):
            a, b = ac.read_list_ints()
            rest = [0] + [pow(i, b, mod) for i in range(1, mod)]
            ans = sum(rest) * (a // mod) + sum(rest[:a % mod + 1])
            ac.st(ans % mod)
        return

    @staticmethod
    def lg_p1939(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1939
        tag: matrix_fast_power
        """

        mat = [[1, 0, 1], [1, 0, 0], [0, 1, 0]]
        lst = [1, 1, 1]
        mod = 10 ** 9 + 7
        mfp = MatrixFastPower()
        for _ in range(ac.read_int()):
            n = ac.read_int()
            if n > 3:
                nex = mfp.matrix_pow(mat, n - 3)
                ans = sum(nex[0]) % mod
                ac.st(ans)
            else:
                ac.st(lst[n - 1])
        return

    @staticmethod
    def lg_p3509(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3509
        tag: two_pointers|implemention|fast_power|sparse_matrix_fast_power|classical|hard|multiplication_method
        """

        n, k, m = ac.read_list_ints()
        nums = ac.read_list_ints()

        ans = list(range(n))
        nex = [0] * n
        head = 0
        tail = k
        for i in range(n):
            while tail + 1 < n and nums[tail + 1] - nums[i] < nums[i] - nums[head]:
                head += 1
                tail += 1
            if nums[tail] - nums[i] <= nums[i] - nums[head]:
                nex[i] = head
            else:
                nex[i] = tail

        while m:
            if m & 1:
                ans = [nex[ans[i]] for i in range(n)]
            nex = [nex[nex[i]] for i in range(n)]
            m >>= 1
        ac.lst([a + 1 for a in ans])
        return

    @staticmethod
    def lg_p1349(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1349
        tag: matrix_fast_power
        """
        p, q, a1, a2, n, m = ac.read_list_ints()
        if n == 1:
            ac.st(a1 % m)
            return
        if n == 2:
            ac.st(a2 % m)
            return
        mat = [[p, q], [1, 0]]
        res = MatrixFastPower().matrix_pow(mat, n - 2, m)
        ans = res[0][0] * a2 + res[0][1] * a1
        ans %= m
        ac.st(ans)
        return

    @staticmethod
    def lg_p2233(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2233
        tag: matrix_fast_power
        """
        n = ac.read_int()
        mat = [[0, 1, 0, 0, 0, 0, 0, 1],
               [1, 0, 1, 0, 0, 0, 0, 0],
               [0, 1, 0, 1, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1, 0, 1],
               [1, 0, 0, 0, 0, 0, 1, 0]]
        res = [1, 0, 0, 0, 0, 0, 0, 0]
        mat_pow = MatrixFastPower().matrix_pow(mat, n - 1, 1000)
        ans = [sum(mat_pow[i][j] * res[j] for j in range(8)) for i in range(8)]
        final = (ans[3] + ans[5]) % 1000
        ac.st(final)
        return

    @staticmethod
    def lg_p2613(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2613
        tag: mod_reverse
        """
        mod = 19260817
        a = ac.read_int()
        b = ac.read_int()
        ans = a * pow(b, -1, mod)
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p3758(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3758
        tag: matrix_dp|matrix_fast_power
        """

        n, m = ac.read_list_ints()
        grid = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            grid[i][i] = 1
            grid[0][i] = 1
        for _ in range(m):
            u, v = ac.read_list_ints()
            grid[u][v] = grid[v][u] = 1

        initial = [0] * (n + 1)
        initial[1] = 1
        mod = 2017
        t = ac.read_int()
        ans = MatrixFastPower().matrix_pow(grid, t, mod)
        res = 0
        for i in range(n + 1):
            res += sum(ans[i][j] * initial[j] for j in range(n + 1))
            res %= mod
        ac.st(res)
        return

    @staticmethod
    def lg_p5343(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5343
        tag: linear_dp|matrix_fast_power
        """

        mod = 10 ** 9 + 7
        n = ac.read_int()
        ac.read_int()
        a = set(ac.read_list_ints())
        ac.read_int()  # TLE
        pre = sorted(list(a.intersection(set(ac.read_list_ints()))))
        m = max(pre)

        dp = [0] * (m + 1)
        dp[0] = 1
        for i in range(1, m + 1):
            for j in pre:
                if i < j:
                    break
                dp[i] += dp[i - j]
            dp[i] %= mod
        if n <= m:
            ac.st(dp[n])
            return

        mat = [0] * (m + 1) * (m + 1)
        for i in range(m, 0, -1):
            mat[i * (m + 1) + i - 1] = 1
        for j in pre:
            mat[j - 1] = 1

        res = MatrixFastPowerFlatten().matrix_pow_flatten(mat, m + 1, n - m, mod)
        ans = 0
        for j in range(m + 1):
            ans += res[j] * dp[m - j]
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p8557(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8557
        tag: brain_teaser|fast_power|counter|classical|hard
        """
        mod = 998244353
        n, k = ac.read_list_ints()
        ans = pow((pow(2, k, mod) - 1) % mod, n, mod)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8624(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8624
        tag: matrix_dp|matrix_fast_power
        """

        mod = 10 ** 9 + 7
        n, m = ac.read_list_ints()
        rem = [[0] * 6 for _ in range(6)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            rem[i][j] = rem[j][i] = 1
        rev = [3, 4, 5, 0, 1, 2]
        cnt = [1] * 6
        mat = [[0] * 6 for _ in range(6)]
        for i in range(6):
            for j in range(6):
                if not rem[j][rev[i]]:
                    mat[i][j] = 1
        res = MatrixFastPower().matrix_pow(mat, n - 1, mod)
        ans = sum([sum([res[i][j] * cnt[j] for j in range(6)])
                   for i in range(6)])
        ans *= FastPower().fast_power(4, n, mod)
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def ac_27(base, exponent):
        """
        url: https://www.acwing.com/problem/content/26/
        tag: float_fast_power|classical
        """

        if base == 0:
            return 0
        if exponent == 0:
            return 1
        return FastPower().float_fast_pow(base, exponent)

    @staticmethod
    def lg_p1306(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1306
        tag: matrix_fast_power|math|fibonacci_property|classical
        """
        n, m = ac.read_list_ints()
        g = math.gcd(n, m)
        base = [[0, 1], [1, 1]]
        mod = 10 ** 8
        ans = MatrixFastPower().matrix_pow(base, g, mod)
        ac.st(ans[0][1])
        return

    @staticmethod
    def abc_236g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc236/tasks/abc236_g
        tag: matrix_fast_power|matrix_fast_power_min|brain_teaser|classical
        """
        n, t, ll = ac.read_list_ints()
        grid = [[math.inf] * n for _ in range(n)]
        for i in range(1, t + 1):
            u, v = ac.read_list_ints_minus_one()
            grid[v][u] = i
        initial = [0] + [math.inf] * (n - 1)
        res = MatrixFastPowerMin().matrix_pow(grid, ll)
        ans = [max(res[i][0], initial[0]) for i in range(n)]
        ans = [x if x < math.inf else -1 for x in ans]
        ac.lst(ans)
        return

    @staticmethod
    def abc_204f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc204/tasks/abc204_f
        tag: matrix_fast_power|bag_dp|brute_force|build_graph
        """
        m, n = ac.read_list_ints()
        mod = 998244353

        grid = [[0] * (1 << m) for _ in range(1 << m)]
        for s in range(1 << m):
            for row in range(1 << m):
                for col in range(1 << m):
                    for one in range(1 << m):
                        cur = s
                        nex = 0
                        flag = 1
                        for r in range(m):
                            if (row >> r) & 1:
                                if cur & (1 << r) or nex & (1 << r):
                                    flag = 0
                                    break
                                cur |= 1 << r
                                nex |= 1 << r
                        for c in range(m):
                            if (col >> c) & 1:
                                if c == m - 1:
                                    flag = 0
                                    break
                                if cur & (1 << c) or cur & (1 << (c + 1)):
                                    flag = 0
                                    break
                                cur |= 1 << c
                                cur |= 1 << (c + 1)
                        if cur & one:
                            flag = 0
                        if (cur | one) != (1 << m) - 1:
                            flag = 0
                        if flag:
                            grid[nex][s] += 1

        cnt = []
        for s in range(1 << m):
            dp = [0] * (m + 1)
            dp[0] = 1
            for i in range(m):
                if (s >> i) & 1:
                    dp[i + 1] = dp[i]
                else:
                    dp[i + 1] = dp[i]
                    if i - 1 >= 0 and not (s >> (i - 1)) & 1:
                        dp[i + 1] += dp[i - 1]
            cnt.append(dp[-1])

        original = [0] * (1 << m)
        original[-1] = 1
        res = MatrixFastPower().matrix_pow(grid, n, mod)
        ans = 0
        for i in range(1 << m):
            ans += res[i][-1] * original[-1] * cnt[i]
        ac.st(ans % mod)
        return

    @staticmethod
    def abc_199f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc199/tasks/abc199_f
        tag: expectation|matrix_fast_power|classical
        """
        n, m, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        mod = 10 ** 9 + 7
        p = pow(2 * m, -1, mod)

        grid = [[0] * n for _ in range(n)]
        for x in range(n):
            grid[x][x] = 1
            for y in dct[x]:
                grid[x][y] = p
            grid[x][x] -= len(dct[x]) * p
            grid[x][x] %= mod
        res = MatrixFastPower().matrix_pow(grid, k, mod)
        for i in range(n):
            ac.st(sum(res[i][j] * nums[j] for j in range(n)) % mod)
        return

    @staticmethod
    def lc_100472(s: str, t: int, nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/total-characters-in-string-after-transformations-ii/
        tag: matrix_fast_power|classical
        """
        cnt = [0] * 26
        for w in s:
            cnt[ord(w) - ord("a")] += 1
        grid = [[0] * 26 for _ in range(26)]
        for i in range(26):
            for j in range(1, nums[i] + 1):
                grid[(i + j) % 26][i] += 1
        mod = 10 ** 9 + 7
        res = MatrixFastPower().matrix_pow(grid, t, mod)
        ans = [0] * 26
        for i in range(26):
            ans[i] = sum(res[i][j] * cnt[j] for j in range(26))
        return sum(ans) % mod
