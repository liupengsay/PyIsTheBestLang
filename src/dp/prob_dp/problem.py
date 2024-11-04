"""
Algorithm：prob_dp
Description：comb|specific_plan|prob|expectation

====================================LeetCode====================================
1227（https://leetcode.cn/problems/airplane-seat-assignment-probability/）prob_dp

=====================================LuoGu======================================
P1291（https://www.luogu.com.cn/problem/P1291）liner_dp|expectation
P4316（https://www.luogu.com.cn/problem/P4316）expectation|reverse_graph|topological_sort
P6154（https://www.luogu.com.cn/problem/P6154）reverse_graph|expectation|tree_dp|float|mod

=====================================AtCoder======================================
ABC342F（https://atcoder.jp/contests/abc342/tasks/abc342_f）prob_dp
ABC333F（https://atcoder.jp/contests/abc333/tasks/abc333_f）matrix_dp|equation|prob_dp|math|implemention
ABC326E（https://atcoder.jp/contests/abc326/tasks/abc326_e）prob_dp|contribution_method
ABC323E（https://atcoder.jp/contests/abc323/tasks/abc323_e）linear_dp|prob_dp|brute_force|classical
ABC300E（https://atcoder.jp/contests/abc300/tasks/abc300_e）prob_dp|math|classical|brain_teaser
ABC298E（https://atcoder.jp/contests/abc298/tasks/abc298_e）prob_dp
ABC297F（https://atcoder.jp/contests/abc297/tasks/abc297_f）matrix_dp|inclusion_exclusion|prob_dp
ABC280E（https://atcoder.jp/contests/abc280/tasks/abc280_e）prob_dp|expectation_dp|classical
ABC275E（https://atcoder.jp/contests/abc275/tasks/abc275_e）prob_dp|linear_dp|classical
ABC266E（https://atcoder.jp/contests/abc266/tasks/abc266_e）expectation_dp|brain_teaser|classical
ABC263E（https://atcoder.jp/contests/abc263/tasks/abc263_e）expectation_dp|reverse_order|math|brain_teaser|classical
ABC243F（https://atcoder.jp/contests/abc243/tasks/abc243_f）matrix_dp|prob_dp|brain_teaser|comb|math
ABC360E（https://atcoder.jp/contests/abc360/tasks/abc360_e）prob_dp|implemention|math
ABC194D（https://atcoder.jp/contests/abc194/tasks/abc194_d）prob_dp
ABC193D（https://atcoder.jp/contests/abc193/tasks/abc193_d）prob|math
ABC189F（https://atcoder.jp/contests/abc189/tasks/abc189_f）expectation_dp|high_precision|math|reverse_order|suffix_sum_opt|classical

===================================CodeForces===================================
540D（https://codeforces.com/problemset/problem/540/D）prob_dp|bag_dp|math|game_dp
1265E（https://codeforces.com/problemset/problem/1265/E）expectation_dp|math|classical|circle_dp|prob_dp
2020E（https://codeforces.com/contest/2020/problem/E）expectation_dp|implemention|data_range
1753F（https://codeforces.com/problemset/problem/1753/C）expectation_dp|comb|inv|prefix_sum
518D（https://codeforces.com/contest/518/problem/D）bag_dp|expectation_dp|prob_dp
453A（https://codeforces.com/problemset/problem/453/A）expectation_dp|prob_dp|inclusion_exclusion|float_fast_power
442B（https://codeforces.com/problemset/problem/442/B）prob_dp|math|greedy

=====================================AcWing=====================================
5058（https://www.acwing.com/problem/content/description/5061/）prob_dp


"""
import math
from collections import deque

from src.math.comb_perm.template import Combinatorics
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1291(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1291
        tag: liner_dp|expectation
        """
        n = ac.read_int()
        ans = [1, 1]
        for x in range(2, n + 1):
            a, b = ans
            c, d = 1, x
            g = math.gcd(b, d)
            lcm = b * d // g
            a, b = a * lcm // b + c * lcm // d, lcm
            g = math.gcd(a, b)
            ans = [a // g, b // g]
        a, b = ans
        a *= n
        x = a // b
        a %= b
        if a == 0:
            ac.st(x)
            return
        g = math.gcd(a, b)
        ans = [a // g, b // g]
        a, b = ans
        ac.st(len(str(x)) * " " + str(a))
        ac.st(str(x) + "-" * len(str(b)))
        ac.st(len(str(x)) * " " + str(b))
        return

    @staticmethod
    def lg_p4316(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4316
        tag: expectation|reverse_graph|topological_sort|float
        """
        n, m = ac.read_list_ints()
        dp = [0 for _ in range(n)]
        degree = [0] * n
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, w = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[b][a] = w
            degree[a] += 1
        cnt = degree[:]

        stack = deque([n - 1])
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                dp[j] += dp[i] + dct[i][j]
                degree[j] -= 1
                if not degree[j]:
                    dp[j] /= cnt[j]
                    stack.append(j)
        ans = "%.2f" % (dp[0])
        ac.st(ans)
        return

    @staticmethod
    def lg_p6154(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6154
        tag: reverse_graph|expectation|tree_dp|float|mod
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            dct[y].append(x)
            degree[x] += 1
        length_sum = [0] * n
        path_cnt = [0] * n
        mod = 998244353
        stack = deque([i for i in range(n) if not degree[i]])
        for i in stack:
            path_cnt[i] = 1
        while stack:
            i = stack.popleft()
            for j in dct[i]:
                degree[j] -= 1
                length_sum[j] += path_cnt[i] + length_sum[i]
                path_cnt[j] += path_cnt[i]
                if not degree[j]:
                    path_cnt[j] += 1
                    path_cnt[j] %= mod
                    length_sum[j] %= mod
                    stack.append(j)
        total_length = sum(length_sum) % mod
        total_cnt = sum(path_cnt) % mod
        ac.st(total_length * pow(total_cnt, -1, mod) % mod)
        return

    @staticmethod
    def ac_5058(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/5061/
        tag: prob_dp
        """
        w, b = ac.read_list_ints()
        dp = [[0] * (b + 1) for _ in range(w + 1)]
        for i in range(1, w + 1):
            dp[i][0] = 1
        for i in range(1, w + 1):
            for j in range(1, b + 1):
                p = i / (i + j)
                if j > 1:
                    p += j / (i + j) * (j - 1) / (i + j - 1) * i / (i + j - 2) * dp[i - 1][j - 2]
                if j > 2:
                    p += j / (i + j) * (j - 1) / (i + j - 1) * (j - 2) / (i + j - 2) * dp[i][j - 3]
                dp[i][j] = p
        ac.st(dp[w][b])
        return

    @staticmethod
    def abc_333f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc333/tasks/abc333_f
        tag: matrix_dp|equation|prob_dp|math|implemention
        """
        n = ac.read_int()
        mod = 998244353
        pp = [pow(2, i, mod) for i in range(3001)]
        pv = [0 for _ in range(3001)]
        x = 1
        for i in range(1, 3001):
            pv[i] = pow(x, -1, mod)
            x = (x * 2 + 1) % mod
        p1 = pow(2, -1, mod)
        dp = [0] * n
        dp[0] = 1
        for i in range(1, n):
            ndp = [0] * n
            ndp[0] = sum(dp[j] * pp[j] for j in range(i)) * pow(2 ** (i + 1) - 1, -1, mod) % mod
            for j in range(1, i + 1):
                ndp[j] = (p1 * (dp[j - 1] + ndp[j - 1])) % mod
            dp = ndp
        ac.lst([x for x in dp])
        return

    @staticmethod
    def abc_323e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc323/tasks/abc323_e
        tag: linear_dp|prob_dp|brute_force|classical
        """
        n, x = ac.read_list_ints()
        t = ac.read_list_ints()
        dp = [0] * (x + 1)
        dp[0] = 1
        mod = 998244353
        pp = pow(n, -1, mod)
        for i in range(1, x + 1):
            for j in range(n):
                if i >= t[j]:
                    dp[i] += dp[i - t[j]] * pp
            dp[i] %= mod
        res = 0
        for i in range(x + 1):
            if i + t[0] > x:
                res += dp[i]
        res = (res * pp) % mod
        ac.st(res)
        return

    @staticmethod
    def abc_280e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc280/tasks/abc280_e
        tag: prob_dp|expectation_dp|classical
        """
        n, p = ac.read_list_ints()
        mod = 998244353
        p2 = p * pow(100, -1, mod) % mod
        p1 = (100 - p) * pow(100, -1, mod) % mod
        dp = [0] * (n + 2)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = (dp[i - 1] * p1 + dp[i - 2] * p2 + 1) % mod
        ac.st(dp[n])
        return

    @staticmethod
    def abc_275e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc275/tasks/abc275_e
        tag: prob_dp|linear_dp|classical
        """
        mod = 998244353
        n, m, k = ac.read_list_ints()
        dp = [0] * (n + 1)
        dp[0] = 1
        ans = 0
        p = pow(m, -1, mod)
        for _ in range(k):
            ndp = [0] * (n + 1)
            for i in range(n):
                for j in range(1, m + 1):
                    x = i + j
                    if x > n:
                        x = n - (x - n)
                    ndp[x] += dp[i] * p % mod
            dp = [x % mod for x in ndp]
            ans += dp[-1]
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def abc_266e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc266/tasks/abc266_e
        tag: expectation_dp|brain_teaser|classical
        """
        n = ac.read_int()
        pre = 0
        for i in range(1, n + 1):
            cur = 0
            for j in range(1, 6 + 1):
                if j > pre:
                    cur += j
                else:
                    cur += pre
            pre = cur / 6
        ac.st(pre)
        return

    @staticmethod
    def abc_263e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc263/tasks/abc263_e
        tag: expectation_dp|reverse_order|math|brain_teaser|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        post = [0] * (n + 1)
        mod = 998244353
        cb = Combinatorics(n + 10, mod)
        dp = [0] * n
        for i in range(n - 2, -1, -1):
            x = min(n - 1, i + nums[i])
            p = post[i + 1] - post[x + 1]
            dp[i] = (p + nums[i] + 1) * cb.inv[nums[i]]
            dp[i] %= mod
            post[i] = (post[i + 1] + dp[i]) % mod
        ac.st(dp[0])
        return

    @staticmethod
    def abc_243f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc243/tasks/abc243_f
        tag: matrix_dp|prob_dp|brain_teaser|comb|math
        """
        mod = 998244353
        cb = Combinatorics(50, mod)

        n, m, k = ac.read_list_ints()
        w = [ac.read_int() for _ in range(n)]
        tot = sum(w)
        p = pow(tot, -1, mod)
        pp = [ww * p % mod for ww in w]
        dp = [[0] * (m + 1) for _ in range(k + 1)]
        dp[k][m] = 1
        for i in range(n - 1, -1, -1):
            ndp = [[0] * (m + 1) for _ in range(k + 1)]
            for j in range(k + 1):
                for x in range(m + 1):
                    res = dp[j][x]
                    if x + 1 <= m:
                        for c in range(1, k - j + 1):
                            res += dp[j + c][x + 1] * cb.comb(j + c, c) * pow(pp[i], c, mod)
                    ndp[j][x] = res % mod
            dp = [[x % mod for x in ls] for ls in ndp]
        ac.st(dp[0][0])
        return

    @staticmethod
    def abc_360e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc360/tasks/abc360_e
        tag: prob_dp|implemention|math
        """
        n, k = ac.read_list_ints()

        if n == 1:
            ac.st(1)
            return
        mod = 998244353
        nn = pow(n * n, -1, mod)
        pp = pow(n - 1, -1, mod)
        a = (n * n - 2 * n) * nn % mod
        b = 2 * nn % mod
        ak = pow(a, k, mod)
        one = (ak + b * (ak - 1) * pow(a - 1, -1, mod)) % mod
        zero = ((1 - one) * pp) % mod
        ans = (one + zero * (n * (n + 1) // 2 - 1)) % mod
        ac.st(ans)
        return

    @staticmethod
    def cf_540d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/540/D
        tag: prob_dp|bag_dp|math|game_dp
        """

        r, s, p = ac.read_list_ints()
        n = r + s + p

        dp = [[[0, 0, 0] for _ in range(s + 1)] for _ in range(r + 1)]

        for i in range(n - 1, -1, -1):
            for a in range(min(r, n - i), -1, -1):
                for b in range(min(s, n - i - a), -1, -1):
                    c = n - i - a - b
                    if a == b == 0:
                        dp[a][b] = [0, 0, 1]
                    elif b == c == 0:
                        dp[a][b] = [1, 0, 0]
                    elif a == c == 0:
                        dp[a][b] = [0, 1, 0]
                    else:
                        prob = a * b + b * c + c * a
                        res = [0, 0, 0]
                        if a and b:
                            nex = dp[a][b - 1]
                            for j in range(3):
                                res[j] += a * b * nex[j] / prob
                        if a and c:
                            nex = dp[a - 1][b]
                            for j in range(3):
                                res[j] += a * c * nex[j] / prob
                        if b and c:
                            nex = dp[a][b]
                            for j in range(3):
                                res[j] += c * b * nex[j] / prob
                        dp[a][b] = res
        ans = dp[r][s]
        tot = sum(ans)
        ac.lst([x / tot for x in ans])
        return

    @staticmethod
    def cf_1265e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1265/E
        tag: expectation_dp|math|classical|circle_dp|prob_dp
        """
        mod = 998244353
        ac.read_int()
        rev = [pow(x, -1, mod) for x in range(1, 101)]
        ans = 0
        p = ac.read_list_ints_minus_one()
        for x in p:
            ans = (ans + 1) * 100 * rev[x] % mod
        ac.st(ans)
        return

    @staticmethod
    def cf_1753f(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1753/C
        tag: expectation_dp|comb|inv|prefix_sum
        """
        mod = 998244353
        cb = Combinatorics(2 * 10 ** 5, mod)
        lst = [x * x % mod for x in cb.inv[1:]]
        pre = ac.accumulate(lst)
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            cnt = sum(nums)
            x = sum(nums[:n - cnt])
            if x == 0:
                ac.st(0)
                continue
            ans = pre[x] * n * (n - 1) // 2
            ac.st(ans % mod)
        return

    @staticmethod
    def cf_453a(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/453/A
        tag: expectation_dp|prob_dp|inclusion_exclusion|float_fast_power
        """
        m, n = ac.read_list_ints()
        dp = [0] * (m + 1)
        for x in range(1, m + 1):
            dp[x] = pow(x / m, n)
        ans = sum((dp[i] - dp[i - 1]) * i for i in range(1, m + 1))
        ac.st(ans)
        return

    @staticmethod
    def abc_189f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc189/tasks/abc189_f
        tag: expectation_dp|high_precision|math|reverse_order|suffix_sum_opt|classical
        """
        n, m, k = ac.read_list_ints()
        dpa = [0] * n
        dpb = [0] * n
        for i in ac.read_list_ints():
            dpa[i] = 1
        post = [0, 0]
        for i in range(n - 1, -1, -1):
            if not dpa[i]:
                dpa[i] = post[0] / m
                dpb[i] = post[1] / m + 1
            post[0] += dpa[i]
            post[1] += dpb[i]
            if i + m < n:
                post[0] -= dpa[i + m]
                post[1] -= dpb[i + m]
        if abs(1 - dpa[0]) > 1e-6:
            ans = dpb[0] / (1 - dpa[0])
            ac.st(ans)
        else:
            ac.st(-1)
        return
