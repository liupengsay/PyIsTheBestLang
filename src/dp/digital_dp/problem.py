"""
Algorithm：digital_dp
Description：lexicographical_order|counter|high_to_low|low_to_high


====================================LeetCode====================================
233（https://leetcode.cn/problems/number-of-digit-one/）counter|digital_dp
357（https://leetcode.cn/problems/count-numbers-with-unique-digits/）comb|digital_dp
600（https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/）counter|digital_dp
902（https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/）counter|digital_dp
1012（https://leetcode.cn/problems/numbers-with-repeated-digits/）inclusion_exclusion|counter|digital_dp
1067（https://leetcode.cn/problems/digit-count-in-range/）counter|digital_dp|inclusion_exclusion
1397（https://leetcode.cn/problems/find-all-good-strings/）digital_dp|implemention|kmp
2376（https://leetcode.cn/problems/count-special-integers/）counter|digital_dp
2719（https://leetcode.cn/problems/count-of-integers/）digital_dp|inclusion_exclusion
2801（https://leetcode.cn/problems/count-stepping-numbers-in-range/）digital_dp|inclusion_exclusion
2827（https://leetcode.cn/problems/number-of-beautiful-integers-in-the-range/）digital_dp|inclusion_exclusion
17（https://leetcode.cn/problems/number-of-2s-in-range-lcci/）counter|digital_dp
3352（https://leetcode.cn/problems/count-k-reducible-numbers-less-than-n/description/）digital_dp|linear_dp|preprocess|observation|data_range

100160（https://leetcode.cn/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/）bit_operation|binary_search|bit_operation|binary_search|digital_dp
104301C（https://codeforces.com/gym/104301/problem/C）digital_dp

====================================AtCoder=====================================
ABC121D（https://atcoder.jp/contests/abc121/tasks/abc121_d）xor_property|digital_dp
ABC208E（https://atcoder.jp/contests/abc208/tasks/abc208_e）brain_teaser|digital_dp
ABC336E（https://atcoder.jp/contests/abc336/tasks/abc336_e）brute_force|digital_dp
ABC317F（https://atcoder.jp/contests/abc317/tasks/abc317_f）2-base|digital_dp|classical
ABC295F（https://atcoder.jp/contests/abc295/tasks/abc295_f）digital_dp|kmp_automaton|classical
ABC235F（https://atcoder.jp/contests/abc235/tasks/abc235_f）digital_dp|classical

===================================CodeForces===================================
628D（https://codeforces.com/problemset/problem/628/D）digital_dp
55D（https://codeforces.com/contest/55/problem/D）digital_dp|memorized|classical
914C（https://codeforces.com/problemset/problem/914/C）digital_dp|linear_dp|corner_case

=====================================LuoGu======================================
P1590（https://www.luogu.com.cn/problem/P1590）counter|digital_dp
P1239（https://www.luogu.com.cn/problem/P1239）counter|digital_dp
P3908（https://www.luogu.com.cn/problem/P3908）xor_property|digital_dp|counter|odd_even
P1836（https://www.luogu.com.cn/problem/P1836）digital_dp

======================================Other======================================
（https://www.lanqiao.cn/problems/5891/learning/?contest_id=145）inclusion_exclusion|digital_dp

"""
from functools import lru_cache

from src.dp.digital_dp.template import DigitalDP
from src.string.kmp.template import KMP
from src.util.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_121d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc121/tasks/abc121_d
        tag: xor_property|digital_dp
        """

        #  n^(n+1) == 1 (n%2==0)
        def count(num):
            @lru_cache(None)
            def dfs(i, cnt, is_limit, is_num):
                if i == n:
                    if is_num:
                        return cnt
                    return 0
                res = 0
                if not is_num:
                    res += dfs(i + 1, 0, False, False)

                floor = 0 if is_num else 1
                ceil = int(s[i]) if is_limit else 9
                for x in range(floor, ceil + 1):
                    res += dfs(i + 1, cnt + int(i == d and x == 1),
                               is_limit and ceil == x, True)
                return res

            if num <= 0:
                return 0
            s = bin(num)[2:]
            n = len(s)
            ans = 0
            for d in range(n):
                c = dfs(0, 0, True, False)
                dfs.cache_clear()
                if c % 2:
                    ans += 1 << (n - d - 1)
            return ans

        a, b = ac.read_list_ints()
        ac.st(count(b) ^ count(a - 1))
        return

    @staticmethod
    def abc_208e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc208/tasks/abc208_e
        tag: brain_teaser|digital_dp
        """

        @lru_cache(None)
        def dfs(i, is_limit, is_num, pre):
            if i == m:
                return int(is_num) and pre <= k
            res = 0
            if not is_num:
                res += dfs(i + 1, False, False, 0)
            low = 0 if is_num else 1
            high = int(st[i]) if is_limit else 9
            for x in range(low, high + 1):
                y = pre * x if is_num else x
                if y > k:
                    y = k + 1
                res += dfs(i + 1, is_limit and high == x, True, y)
            return res

        n, k = ac.read_list_ints()
        st = str(n)
        m = len(st)
        ans = dfs(0, True, False, 0)
        ac.st(ans)
        return

    @staticmethod
    def lc_233(n: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-digit-one/
        tag: counter|digital_dp
        """
        return DigitalDP().count_digit_dp(n, 1)

    @staticmethod
    def lc_2719_1(num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        """
        url: https://leetcode.cn/problems/count-of-integers/
        tag: digital_dp|inclusion_exclusion
        """

        def check(n):
            # calculate the number of occurrences of positive integer binary bit 1 from 1 to n

            @lru_cache(None)
            def dfs(i, is_limit, is_num, cnt):
                if i == m:
                    return 1 if (min_sum <= cnt <= max_sum and is_num) else 0
                if cnt + 9 * (m - i) < min_sum:
                    return 0
                if cnt > max_sum:
                    return 0
                res = 0
                if not is_num:
                    res += dfs(i + 1, False, False, 0)
                low = 0 if is_num else 1
                high = int(st[i]) if is_limit else 9
                for x in range(low, high + 1):
                    if cnt + x <= max_sum:
                        res += dfs(i + 1, is_limit and high == x, True, cnt + x)
                return res % mod

            st = str(n)
            m = len(st)
            ans = dfs(0, True, False, 0)
            dfs.cache_clear()
            return ans

        mod = 10 ** 9 + 7
        return (check(int(num2)) - check(int(num1) - 1)) % mod

    @staticmethod
    def lc_2719_2(num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        """
        url: https://leetcode.cn/problems/count-of-integers/
        tag: digital_dp|inclusion_exclusion
        """
        mod = 10 ** 9 + 7

        def dfs(i, down, up, pre, is_num):
            if i < 0:
                return 1 if is_num and min_sum <= pre <= max_sum else 0
            if not down and not up and dp[i][pre][is_num] > -1:
                return dp[i][pre][is_num]

            res = 0
            if pre + 9 * (i + 1) >= min_sum:
                floor = int(low[i]) if down else (0 if is_num else 1)
                ceil = int(high[i]) if up else 9
                for x in range(floor, ceil + 1):
                    if pre + x <= max_sum:
                        res += dfs(i - 1, down and x == floor, up and x == ceil, pre + x, 1 if x > 0 or is_num else -1)
                        res %= mod
                    else:
                        break

            if not down and not up:
                dp[i][pre][is_num] = res
            return res

        high = num2[::-1]
        low = num1[::-1]
        low += (len(high) - len(low)) * "0"
        dp = [[[-1] * 2 for _ in range(max_sum + 1)] for _ in range(len(high))]
        ans = dfs(len(high) - 1, True, True, 0, 0)
        return ans

    @staticmethod
    def lc_2801_1(low: str, high: str) -> int:
        """
        url: https://leetcode.cn/problems/count-stepping-numbers-in-range/
        tag: digital_dp|inclusion_exclusion
        """

        def check(num):
            @lru_cache(None)
            def dfs(i, is_limit, is_num, pre):
                if i == n:
                    return is_num
                res = 0
                if not is_num:
                    res += dfs(i + 1, False, 0, -1)

                floor = 0 if is_num else 1
                ceil = int(s[i]) if is_limit else 9
                for x in range(floor, ceil + 1):
                    if pre == -1 or abs(x - pre) == 1:
                        res += dfs(i + 1, is_limit and ceil == x, 1, x)
                return res

            s = str(num)
            n = len(s)
            return dfs(0, True, 0, -1)

        mod = 10 ** 9 + 7
        return (check(int(high)) - check(int(low) - 1)) % mod

    @staticmethod
    def lc_2801_2(low: str, high: str) -> int:
        """
        url: https://leetcode.cn/problems/count-stepping-numbers-in-range/
        tag: digital_dp|inclusion_exclusion
        """
        dp = [[[-1] * 2 for _ in range(11)] for _ in range(101)]
        mod = 10 ** 9 + 7

        def dfs(i, down, up, pre, is_num):
            if i < 0:
                return is_num
            if not down and not up and dp[i][pre][is_num] > -1:
                return dp[i][pre][is_num]
            res = 0

            floor = int(low[i]) if down else (0 if is_num else 1)
            ceil = int(high[i]) if up else 9
            if is_num:
                for x in [pre - 1, pre + 1]:
                    if floor <= x <= ceil:
                        res += dfs(i - 1, down and x == floor, up and x == ceil, x, is_num)
                        res %= mod
            else:
                for x in range(floor, ceil + 1):
                    res += dfs(i - 1, down and x == floor, up and x == ceil, x, int(x > 0))
                    res %= mod
            if not down and not up:
                dp[i][pre][is_num] = res
            return res

        high = high[::-1]
        low = low[::-1]
        low += (len(high) - len(low)) * "0"
        ans = dfs(len(high) - 1, True, True, -1, 0)
        return ans

    @staticmethod
    def lc_2827_1(low: int, high: int, k: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-beautiful-integers-in-the-range/
        tag: digital_dp|inclusion_exclusion
        """

        def check(num):
            @lru_cache(None)
            def dfs(i, is_limit, is_num, odd, rest):
                if i == n:
                    return 1 if is_num and not odd and not rest else 0
                res = 0
                if not is_num:
                    res += dfs(i + 1, False, 0, 0, 0)

                floor = 0 if is_num else 1
                ceil = int(s[i]) if is_limit else 9
                for x in range(floor, ceil + 1):
                    res += dfs(i + 1, is_limit and ceil == x, 1, odd + 1 if x % 2 == 0 else odd - 1,
                               (rest * 10 + x) % k)
                return res

            s = str(num)
            n = len(s)
            return dfs(0, True, 0, 0, 0)

        return check(high) - check(low - 1)

    @staticmethod
    def lc_2827_2(low: int, high: int, k: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-beautiful-integers-in-the-range/
        tag: digital_dp|inclusion_exclusion
        """
        dp = [[[[[-1] * 2 for _ in range(21)] for _ in range(21)] for _ in range(25)] for _ in range(11)]

        def dfs(i, down, up, odd, rest, is_num):
            if i < 0:
                return 1 if is_num and not odd and not rest else 0

            if not down and not up and dp[i][odd][k][rest][is_num] > -1:
                return dp[i][odd][k][rest][is_num]

            res = 0
            floor = int(low[i]) if down else (0 if is_num else 1)
            ceil = int(high[i]) if up else 9
            for x in range(floor, ceil + 1):
                cur_is_num = 1 if x > 0 or is_num else 0
                res += dfs(i - 1, down and x == floor, up and x == ceil,
                           0 if not cur_is_num else odd + 1 if x % 2 == 0 else odd - 1,
                           (rest * 10 + x) % k, cur_is_num)

            if not down and not up:
                dp[i][odd][k][rest][is_num] = res
            return res

        high = str(high)[::-1]
        low = str(low)[::-1]
        low += (len(high) - len(low)) * "0"
        ans = dfs(len(high) - 1, True, True, 0, 0, 0)
        return ans

    @staticmethod
    def lg_p1836(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1836
        tag: digital_dp
        """
        n = ac.read_int()
        ac.st(DigitalDP().count_digit_sum(n))
        return

    @staticmethod
    def lc_1067(d: int, low: int, high: int) -> int:
        """
        url: https://leetcode.cn/problems/digit-count-in-range/
        tag: counter|digital_dp|inclusion_exclusion
        """
        dd = DigitalDP()
        return dd.count_digit_dp(high, d) - dd.count_digit_dp(low - 1, d)

    @staticmethod
    def abc_336e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc336/tasks/abc336_e
        tag: brute_force|digital_dp
        """
        num = ac.read_int()
        num += 1
        lst = [int(x) for x in str(num)]
        n = len(lst)
        ans = 0
        for digit_sum in range(1, 9 * n + 1):
            pre = [0] * digit_sum * (digit_sum + 1)
            x = x_mod = 0
            for k in range(n):
                cur = [0] * digit_sum * (digit_sum + 1)
                for i in range(min(digit_sum + 1, (k + 1) * 9 + 1)):
                    for j in range(digit_sum):
                        if not pre[i * digit_sum + j]:
                            continue
                        for d in range(10):
                            if i + d > digit_sum:
                                break
                            cur[(i + d) * digit_sum + (j * 10 + d) % digit_sum] += pre[i * digit_sum + j]
                for i in range(lst[k]):
                    if x + i <= digit_sum:
                        cur[(x + i) * digit_sum + (x_mod * 10 + i) % digit_sum] += 1
                x += lst[k]
                x_mod = (x_mod * 10 + lst[k]) % digit_sum
                pre = cur
            ans += pre[digit_sum * digit_sum + 0]
        ac.st(ans)
        return

    @staticmethod
    def abc_317f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc317/tasks/abc317_f
        tag: 2-base|digital_dp|classical
        """

        n, a, b, c = ac.read_list_ints()
        lst = [int(w) for w in bin(n)[2:]]
        m = len(lst)

        mod = 998244353
        tmp = [(0, 1, 1), (1, 0, 1), (0, 0, 0), (1, 1, 0)]

        @lru_cache(None)
        def dfs(i, x, y, z, is_x, is_y, is_z, num_x, num_y, num_z):
            if i == m:
                return x == y == z == 0 and num_x == num_y == num_z == 1
            res = 0
            for xx, yy, zz in tmp:
                if (not is_x or xx <= lst[i]) and (not is_y or yy <= lst[i]) and (not is_z or zz <= lst[i]):
                    nex_x = (x * 2 + xx) % a
                    nex_y = (y * 2 + yy) % b
                    nex_z = (z * 2 + zz) % c
                    nex_is_x = int(is_x and xx == lst[i])
                    nex_is_y = int(is_y and yy == lst[i])
                    nex_is_z = int(is_z and zz == lst[i])
                    res += dfs(i + 1, nex_x, nex_y, nex_z, nex_is_x, nex_is_y, nex_is_z, num_x | xx, num_y | yy,
                               num_z | zz)
            return res % mod

        ans = dfs(0, 0, 0, 0, 1, 1, 1, 0, 0, 0)
        ac.st(ans)
        return

    @staticmethod
    def abc_295f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc295/tasks/abc295_f
        tag: digital_dp|kmp_automaton|classical
        """

        def check(num):

            @lru_cache(None)
            def dfs(i, is_limit, is_num, cnt, pre):
                if i == m:

                    if is_num:
                        return cnt
                    return 0
                res = 0
                if not is_num:
                    res += dfs(i + 1, False, False, 0, 0)
                low = 0 if is_num else 1
                high = int(st[i]) if is_limit else 9
                for x in range(low, high + 1):
                    nex_length = nex[pre * 10 + x]
                    nex_cnt = cnt + 1 if nex_length == n else cnt
                    res += dfs(i + 1, is_limit and high == x, True, nex_cnt, nex_length)
                return res

            st = str(num)
            m = len(st)
            ans = dfs(0, True, False, 0, 0)
            dfs.cache_clear()
            return ans

        for _ in range(ac.read_int()):
            s, ll, rr = ac.read_list_strs()
            lst = [int(w) for w in s]
            n = len(s)
            nex = KMP().kmp_automaton(lst, 10)
            cur = check(int(rr)) - check(int(ll) - 1)
            ac.st(cur)
        return

    @staticmethod
    def abc_235f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc235/tasks/abc235_f
        tag: digital_dp|classical
        """

        mod = 998244353  # TLE
        s = ac.read_str()
        ac.read_int()
        c = ac.read_list_ints()
        target = sum(1 << x for x in c)
        k = len(s)

        weight = [[0] * 10 for _ in range(k + 1)]
        for x in range(10):
            weight[1][x] = x % mod
        for ii in range(2, k + 1):
            for x in range(10):
                weight[ii][x] = (weight[ii - 1][x] * 10) % mod

        @lru_cache(None)
        def dfs(i, state, is_limit, is_num):
            if i == k:
                return (1, 0) if state & target == target else (0, -1)
            res = -1
            tot = -1
            if not is_num:
                cc, ss = dfs(i + 1, 0, False, False)
                if ss > -1:
                    if res == -1:
                        res = ss
                        tot = cc
                    else:
                        res += ss
                        tot += cc

            floor = 0 if is_num else 1
            ceil = int(s[i]) if is_limit else 9
            for x in range(floor, ceil + 1):
                cc, ss = dfs(i + 1, state | (1 << x), is_limit and ceil == x, True)
                if ss > -1:
                    if res == -1:
                        res = ss + weight[k - i][x] * cc
                        tot = cc
                    else:
                        res += ss + weight[k - i][x] * cc
                        tot += cc
            return (tot % mod, res % mod) if res > -1 else (0, -1)

        ans = dfs(0, 0, 1, 0)[1]
        ac.st(ans if ans > -1 else 0)
        return

    @staticmethod
    def cf_628d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/628/D
        tag: digital_dp
        """
        m, d = ac.read_list_ints()  # TLE
        a = [int(w) for w in ac.read_str()]
        b = ac.read_str()
        k = len(a)
        for i in range(k - 1, -1, -1):
            if a[i]:
                a[i] -= 1
                for j in range(i + 1, k):
                    a[j] = 9
                break
        if len(a) >= 2 and a[0] == 0:
            a.pop(0)
        a = "".join(str(x) for x in a)
        mod = 10 ** 9 + 7

        def count(s):
            n = len(s)
            dp = [0] * m * (1 << 3)
            ndp = [0] * m * (1 << 3)
            for p in range(n, -1, -1):
                for rest in range(m):
                    for state in range(1 << 3):
                        is_limit = (state >> 2) & 1
                        is_num = (state >> 1) & 1
                        odd = state & 1
                        if p == n:
                            ndp[state * m + rest] = int(rest == 0 and is_num)
                            continue
                        else:
                            res = 0
                            if not is_num:
                                res += dp[0]

                            floor = 0 if is_num else 1
                            ceil = int(s[p]) if is_limit else 9
                            if odd == 1:
                                if floor <= d <= ceil:
                                    res += dp[
                                        ((int(is_limit and ceil == d) << 2) | 2 | (odd ^ 1)) * m + (rest * 10 + d) % m]
                                ndp[state * m + rest] = res % mod
                                continue
                            for x in range(floor, ceil + 1):
                                if odd == 0 and x == d:
                                    continue
                                res += dp[
                                    ((int(is_limit and ceil == x) << 2) | 2 | (odd ^ 1)) * m + (rest * 10 + x) % m]
                            ndp[state * m + rest] = res % mod
                for rest in range(m):
                    for state in range(1 << 3):
                        dp[state * m + rest] = ndp[state * m + rest]
            return dp[4 * m]

        final = count(b) - count(a)
        ac.st(final % mod)
        return

    @staticmethod
    def lc_3352_1(s: str, k: int) -> int:
        """
        url: https://leetcode.cn/problems/count-k-reducible-numbers-less-than-n/description/
        tag: digital_dp|linear_dp|preprocess|observation|data_range
        """

        @lru_cache(None)
        def dfs(i, is_limit, cnt):
            if i == m:
                return dp[cnt] + 1 <= k if cnt else 0
            res = 0
            if not cnt:
                res += dfs(i + 1, False, cnt)
            low = 0 if cnt else 1
            high = int(s[i]) if is_limit else 1
            for x in range(low, high + 1):
                res += dfs(i + 1, is_limit and high == x, cnt + int(x == 1))
            return res % mod

        dp = [0] * 801
        for num in range(2, 801):
            dp[num] = dp[num.bit_count()] + 1

        mod = 10 ** 9 + 7

        m = len(s)
        ans = dfs(0, True, 0)
        dfs.cache_clear()
        if dp[s.count("1")] + 1 <= k:
            ans = (ans - 1) % mod
        return ans

    @staticmethod
    def lc_3352_2(s: str, k: int) -> int:
        """
        url: https://leetcode.cn/problems/count-k-reducible-numbers-less-than-n/description/
        tag: digital_dp|linear_dp|preprocess|observation|data_range
        """
        ceil = 800
        cost = [0] * (ceil + 1)
        for num in range(2, ceil + 1):
            cost[num] = cost[num.bit_count()] + 1

        mod = 10 ** 9 + 7
        n = len(s)
        dp = [[0] for _ in range(2)]
        dp[0][0] = 1
        for c, w in enumerate(s):
            ndp = [[0] * (c + 2) for _ in range(2)]
            for i in range(2):
                for j in range(c + 1):
                    if i == 1:
                        lst = [0, 1]
                    else:
                        lst = [0, 1] if w == "1" else [0]
                    for x in lst:
                        if i == 0 and int(w) == x:
                            ndp[0][j + x] += dp[i][j]
                        else:
                            ndp[1][j + x] += dp[i][j]
            dp = [[x % mod for x in ls] for ls in ndp]
        ans = 0
        for i in range(2):
            for j in range(1, n + 1):
                if cost[j] + 1 <= k:
                    ans += dp[i][j]
                    ans %= mod
        if cost[s.count("1")] + 1 <= k:
            ans = (ans - 1) % mod
        return ans

    @staticmethod
    def cf_914c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/914/C
        tag: digital_dp|linear_dp|corner_case
        """

        ceil = 1000
        cost = [0] * (ceil + 1)
        for num in range(2, ceil + 1):
            cost[num] = cost[num.bit_count()] + 1
        s = ac.read_str()
        k = ac.read_int()
        if k == 0:
            ac.st(1)
            return
        mod = 10 ** 9 + 7
        n = len(s)
        dp = [[0] for _ in range(2)]
        dp[0][0] = 1
        for c, w in enumerate(s):
            ndp = [[0] * (c + 2) for _ in range(2)]
            for i in range(2):
                for j in range(c + 1):
                    if i == 1:
                        lst = [0, 1]
                    else:
                        lst = [0, 1] if w == "1" else [0]
                    for x in lst:
                        if i == 0 and int(w) == x:
                            ndp[0][j + x] += dp[i][j]
                        else:
                            ndp[1][j + x] += dp[i][j]
            dp = [[x % mod for x in ls] for ls in ndp]
        ans = 0
        for i in range(2):
            for j in range(1, n + 1):
                if cost[j] + 1 == k:
                    ans += dp[i][j]
                    ans %= mod
        if k == 1:
            ans = (ans - 1) % mod
        ac.st(ans)
        return
