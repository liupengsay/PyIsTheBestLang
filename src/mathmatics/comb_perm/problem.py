"""

Algorithm：math|comb|counter|multiplicative_reverse|lucas|perm|factorial|rev
Description：combination|permutation|counter|partition_method|fault_perm|stirling_number|catalan_number|inclusion_exclusion
Lucas:（comb(n, m)%p = comb(n%p, m%p)*comb(n//p, m//p)）%p  

====================================LeetCode====================================
96（https://leetcode.com/problems/unique-binary-search-trees/）catalan_number
95（https://leetcode.com/problems/unique-binary-search-trees/）catalan_number|recursion|specific_plan
634（https://leetcode.com/problems/find-the-derangement-of-an-array/）fault_perm|counter|dp
1259（https://leetcode.com/problems/handshakes-that-dont-cross/）catalan_number
2338（https://leetcode.com/problems/count-the-number-of-ideal-arrays/）partition_method|factorization|specific_plan|counter|classical
1735（https://leetcode.com/problems/count-ways-to-make-array-with-product/）prime_factorization|partition_method|classical
1621（https://leetcode.com/problems/number-of-sets-of-k-non-overlapping-line-segments/）partition_method|comb_perm
1866（https://leetcode.com/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/）stirling_number|first_kind_stirling_number
1916（https://leetcode.com/problems/count-ways-to-build-rooms-in-an-ant-colony/）tree_dp|math|comb|counter
D - Blue and Red Balls（https://atcoder.jp/contests/abc132/tasks/abc132_d）comb|math|counter|classical|equation

=====================================LuoGu======================================
4071（https://www.luogu.com.cn/problem/P4071）multiplicative_reverse|comb|perm|recursion|fault_perm
1287（https://www.luogu.com.cn/problem/P1287）second_kind_stirling_number|factorial|dp
1375（https://www.luogu.com.cn/problem/P1375）catalan_number
1754（https://www.luogu.com.cn/problem/P1754）catalan_number
2193（https://www.luogu.com.cn/problem/P2193）partition_method|factorization|comb|specific_plan|classical
1338（https://www.luogu.com.cn/problem/P1338）brute_force|reverse_order_pair
1313（https://www.luogu.com.cn/problem/P1313）math|comb|polynomial
1061（https://www.luogu.com.cn/problem/P1061）implemention|lexicographical_order|nex_perm
3197（https://www.luogu.com.cn/problem/P3197）counter|fast_power|inclusion_exclusion
3414（https://www.luogu.com.cn/problem/P3414）comb|odd_even|fast_power
4369（https://www.luogu.com.cn/problem/P4369）brain_teaser|comb|construction
5520（https://www.luogu.com.cn/problem/P5520）partition_method|comb
3807（https://www.luogu.com.cn/problem/P3807）lucas
1044（https://www.luogu.com.cn/problem/P1044）catalan_number
1655（https://www.luogu.com.cn/problem/P1655）matrix_dp|stirling_number
1680（https://www.luogu.com.cn/problem/P1680）partition_method|multiplicative_reverse|lucas|comb(a,b)%m
2265（https://www.luogu.com.cn/problem/P2265）comb|comb(n+m, m)
2638（https://www.luogu.com.cn/problem/P2638）partition_method|specific_plan|classical
2822（https://www.luogu.com.cn/problem/P2822）counter|comb(i, j) % k == 0
3223（https://www.luogu.com.cn/problem/P3223）inclusion_exclusion|partition_method
3904（https://www.luogu.com.cn/problem/P3904）second_stirling_number|dp|classical
4071（https://www.luogu.com.cn/problem/P4071）fault_perm|specific_plan|counter|classical
5684（https://www.luogu.com.cn/problem/P5684）inclusion_exclusion|counter
6057（https://www.luogu.com.cn/problem/P6057）inclusion_exclusion|counter

===================================CodeForces===================================
1795D（https://codeforces.com/problemset/problem/1795/D）comb|counter|mod|multiplicative_reverse
300C（https://codeforces.com/problemset/problem/300/C）brute_force|comb|specific_plan|counter
559C（https://codeforces.com/problemset/problem/559/C）inclusion_exclusion|counter
1436C（https://codeforces.com/problemset/problem/1436/C）binary_search|comb
414B（https://codeforces.com/problemset/problem/414/B）min_prime|partition_method|counter|dp
1879C（https://codeforces.com/contest/1879/problem/C）greedy|brute_force|comb|counter


====================================AtCoder=====================================
ARC058B（https://atcoder.jp/contests/abc042/tasks/arc058_b）inclusion_exclusion|comb|counter
ARC077B（https://atcoder.jp/contests/abc066/tasks/arc077_b）inclusion_exclusion|comb|counter
ABC110D（https://atcoder.jp/contests/abc110/tasks/abc110_d）prime_factorization|partition_method|counter
ABC127E（https://atcoder.jp/contests/abc127/tasks/abc127_e）contribution_method|comb|counter

=====================================AcWing=====================================
130（https://www.acwing.com/problem/content/132/）catalan_number
4002（https://www.acwing.com/problem/content/4005/）matrix_dp|partition_method|classical
4496（https://www.acwing.com/problem/content/4499/）partition_method|counter
5055（https://www.acwing.com/problem/content/5058/）math|comb|mod


catalan_number（https://oi-wiki.org/math/combinatorics/catalan/）
"""
import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import List

from src.mathmatics.comb_perm.template import Combinatorics, Lucas
from src.mathmatics.number_theory.template import NumberTheory
from src.mathmatics.prime_factor.template import PrimeFactor
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def abc_110d(ac=FastIO()):
        # prime_factorization|与partition_methodcounter
        n, m = ac.read_list_ints()
        mod = 10 ** 9 + 7
        cb = Combinatorics(n + 100, mod)  # 注意这里会超出n
        ans = 1
        for _, c in NumberTheory().get_prime_factor(m):
            ans *= cb.comb(c + n - 1, n - 1)  # n个正整数和为c+n转换
            # 等价于sum(cb.comb(n, k)*cb.comb(c-1, k-1) for k in range(1, c+1))
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def cf_1436c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1436/C
        tag: binary_search|comb
        """

        # binary_search|组合数
        n, x, pos = ac.read_list_ints()
        big = small = 0

        left = 0
        right = n
        while left < right:
            mid = (left + right) // 2
            if mid <= pos:
                small += int(mid != pos)
                left = mid + 1
            else:
                right = mid
                big += 1

        if small >= x or big > n - x:
            ac.st(0)
            return
        mod = 10 ** 9 + 7
        comb = Combinatorics(n, mod)
        ans = comb.comb(n - x, big) * comb.factorial(big) * math.comb(x - 1, small) * math.factorial(small)
        ans *= comb.factorial(n - big - small - 1)
        ac.st(ans % mod)
        return

    @staticmethod
    def cf_559c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/559/C
        tag: inclusion_exclusion|counter
        """

        # inclusion_exclusion组合数
        m, n, k = ac.read_list_ints()
        mod = 10 ** 9 + 7
        cb = Combinatorics(m + n, mod)
        pos = [ac.read_list_ints_minus_one() for _ in range(k)]
        pos.sort()

        def dist(x1, y1, x2, y2):
            return cb.comb(x2 + y2 - x1 - y1, x2 - x1)

        ans = dist(0, 0, m - 1, n - 1)
        bad = []
        for x in range(k):
            i, j = pos[x]
            cur = dist(0, 0, i, j)
            for y in range(x):
                a, b = pos[y]
                if b <= j:
                    cur -= dist(a, b, i, j) * bad[y]
                    cur %= mod
            bad.append(cur)
            ans -= cur * dist(i, j, m - 1, n - 1)
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lc_1259_1(num_people: int) -> int:
        """
        url: https://leetcode.com/problems/handshakes-that-dont-cross/
        tag: catalan_number
        """
        # catalan_number
        n = num_people // 2
        if num_people <= 1:
            return 1
        mod = 10 ** 9 + 7
        cm = Combinatorics(2 * n + 2, mod)
        ans = cm.comb(2 * n, n) - cm.comb(2 * n, n - 1)
        return ans % mod

    @staticmethod
    def lc_1259_2(num_people: int) -> int:
        """
        url: https://leetcode.com/problems/handshakes-that-dont-cross/
        tag: catalan_number
        """
        # catalan_number的数组形式
        n = num_people // 2
        mod = 10 ** 9 + 7
        dp = [1] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = sum(dp[j] * dp[i - 1 - j] for j in range(i)) % mod
        return dp[n]

    @staticmethod
    def lg_p1375(ac=FastIO()):
        # catalan_number
        n = ac.read_int()
        mod = 10 ** 9 + 7
        cm = Combinatorics(2 * n + 2, mod)
        ans = cm.comb(2 * n, n) - cm.comb(2 * n, n - 1)
        ac.st(ans % mod)
        return

    @staticmethod
    def lg_p1754(ac=FastIO()):
        # catalan_number
        n = ac.read_int()
        # catalan_number的另一种区间递推形式 dp[i][j] = dp[i-1][j]+dp[i][j-1]
        # 类似题目也有长为 2n 合法的括号匹配数 h(n) = h(n-1)*(4*n-2)//(n+1)
        # 也可以 h(n) = math.comb(2*n, n)//(n+1) 求解
        ans = math.comb(2 * n, n) - math.comb(2 * n, n - 1)
        assert ans == math.comb(2 * n, n) // (n + 1)  # 不需要mod|时可以直接用这个
        ac.st(ans)
        return

    @staticmethod
    def lc_634(n):
        """
        url: https://leetcode.com/problems/find-the-derangement-of-an-array/
        tag: fault_perm|counter|dp
        """
        # 求错位comb数
        mod = 10 ** 9 + 7
        fault = [0, 0, 1, 2]
        for i in range(4, n + 1):
            fault.append((i - 1) * (fault[i - 1] + fault[i - 2]) % mod)
        return fault[n]

    @staticmethod
    def cf_300c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/300/C
        tag: brute_force|comb|specific_plan|counter
        """
        mod = 10 ** 9 + 7
        a, b, n = ac.read_list_ints()
        c = Combinatorics(n + 1, mod)

        dct = set(f"{a}{b}")
        ans = 0
        for i in range(n + 1):
            num = a * i + b * (n - i)
            if all(w in dct for w in str(num)):
                ans += c.comb(n, i)
                ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def cf_1795d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1795/D
        tag: comb|counter|mod|multiplicative_reverse
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        mod = 998244353
        c = Combinatorics(n // 3 + 1, mod)
        ans = 1
        for i in range(0, n - 2, 3):
            lst = nums[i:i + 3]
            ans *= lst.count(min(lst))
            ans %= mod
        ans *= c.comb(n // 3, n // 6)
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p4017(ac=FastIO()):
        # 组合数与fault_perm求解
        mod = 10 ** 9 + 7
        cb = Combinatorics(10 ** 6, mod)
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            if m > n:
                ac.st(0)
                continue
            ans = cb.comb(n, m) * cb.fault[n - m]
            ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def main_p1287(n, r):

        @lru_cache(None)
        def dfs(a, b):
            # stirling_number，把a个球放入b个盒子且不允许空盒的specific_plan数
            if a < b or b < 0:
                return 0
            if a == b:
                return 1
            # 新球单独放新盒子，或者放入已经有的老盒子
            return dfs(a - 1, b - 1) + b * dfs(a - 1, b)

        # 由于是不同的球还要球的全排列
        x = dfs(n, r) * math.factorial(r)
        return x

    @staticmethod
    def lg_p1287(ac=FastIO()):
        # inclusion_exclusioncounter
        n, r = ac.read_list_ints()
        ans = 0
        for k in range(r):
            cur = ((-1) ** k) * math.comb(r, k) * ((r - k) ** n)
            ans += cur
        ac.st(ans)
        return

    @staticmethod
    def lg_p4071(ac=FastIO()):
        # partition_method组合数
        tp, n, m, p = ac.read_list_ints()

        if n < 2 * m - 1:
            ac.st(0)
            return

        ans = 1
        for x in range(n - 2 * m + 2, n - m + 2):
            ans *= x
            ans %= p
        ac.st(ans)
        return

    @staticmethod
    def lg_p3807(ac=FastIO()):
        # Lucas模板题
        for _ in range(ac.read_int()):
            n, m, p = ac.read_list_ints()
            ans = Lucas().lucas_iter(n + m, n, p)
            ac.st(ans)
        return

    @staticmethod
    def abc_42d(ac=FastIO()):
        # inclusion_exclusion组合counter
        mod = 10 ** 9 + 7
        h, w, a, b = ac.read_list_ints()
        cb = Combinatorics(h + w + 2, mod)
        ans = cb.comb(h + w - 2, h - 1)
        for x in range(h - a + 1, h + 1):
            y = b
            cur = cb.comb(x + y - 2, x - 1) * cb.comb(h - x + w - y - 1, h - x)
            ans = (ans - cur) % mod
        ac.st(ans)
        return

    @staticmethod
    def abc_65d(ac=FastIO()):
        # inclusion_exclusion组合counter
        mod = 10 ** 9 + 7
        n = ac.read_int()
        nums = ac.read_list_ints()
        ind = [-1, -1]
        pre = defaultdict(list)
        for i in range(n + 1):
            pre[nums[i]].append(i)
            if len(pre[nums[i]]) == 2:
                ind = pre[nums[i]]
                break
        x = ind[0]
        y = n - ind[-1]
        cb = Combinatorics(n + 10, mod)
        for k in range(1, n + 2):
            ans = cb.comb(n + 1, k)
            if 1 <= k <= x + y + 1:
                ans -= cb.comb(x + y, k - 1)
            ac.st(ans % mod)
        return

    @staticmethod
    def abc_127e(ac=FastIO()):
        # contribution_method组合counter
        mod = 10 ** 9 + 7
        m, n, k = ac.read_list_ints()
        cb = Combinatorics(m * n, mod)
        cnt = cb.comb(m * n - 2, k - 2)
        ans = 0
        for i in range(m):
            for j in range(n):
                up = n * i * (i + 1) // 2 if i else 0
                down = n * (m - 1 - i + 1) * (m - 1 - i) // 2 if i < m - 1 else 0
                left = m * j * (j + 1) // 2 if j else 0
                right = m * (n - 1 - j + 1) * (n - 1 - j) // 2 if j < n - 1 else 0
                ans += cnt * (left + right + up + down)
                ans %= mod
        ans *= pow(2, -1, mod)
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def ac_130(ac=FastIO()):
        # 超大范围的catalan_number h(n) = C(2n, n)//(n+1) = ((n+1)*..*(2*n))//(1*2*..*(n+1))
        n = ac.read_int()
        nt = PrimeFactor(2 * n + 1)
        cnt = defaultdict(int)
        for i in range(1, 2 * n + 1):
            for num, y in nt.prime_factor[i]:
                if i <= n:
                    cnt[num] -= y
                else:
                    cnt[num] += y
        ans = 1
        for w in cnt:
            ans *= w ** cnt[w]
        ac.st(ans // (n + 1))
        return

    @staticmethod
    def lg_p1655(ac=FastIO()):
        # second_stirling_number只能递推（n个不同的球放入m个相同的盒子，不允许为空，stirling_number）
        n = m = 101  # （n个相同的球放入m个不同的盒子，不允许为空，partition_method）
        dp = [[0] * m for _ in range(n)]
        for i in range(1, n):
            dp[i][i] = dp[i][1] = 1
            for j in range(2, i):
                dp[i][j] = dp[i - 1][j - 1] + j * dp[i - 1][j]
        while True:
            lst = ac.read_list_ints()
            if not lst:
                break
            n, m = lst
            ac.st(dp[n][m])
        return

    @staticmethod
    def lg_p1680(ac=FastIO()):
        # partition_method不同分组的个数，multiplicative_reverse与Lucas定理快速Comb(a,b) % m
        # 转换为（n个相同的球放入m个不同的盒子，不允许为空的specific_plan数）
        n, m = ac.read_list_ints()
        n -= sum([ac.read_int() for _ in range(m)])
        m -= 1
        n -= 1
        p = 10 ** 9 + 7
        ans = Lucas().comb(n, m, p)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2265(ac=FastIO()):
        # comb，comb(n+m, m)
        mod = 1000000007
        n, m = ac.read_list_ints()
        ans = Lucas().comb(n + m, m, mod)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2638(ac=FastIO()):
        # partition_method a 个球放入 n 个盒子不要求每个都放也不要求放完的specific_plan数
        n, a, b = ac.read_list_ints()
        ans = math.comb(n + a, n) * math.comb(n + b, n)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2822(ac=FastIO()):
        # 组合数 comb(i, j) % k == 0 的个数
        t, k = ac.read_list_ints()

        # preprocess组合数  comb(i, j) % k
        x = 2001
        dp = [[0] * x for _ in range(x)]
        dp[0][0] = 1
        for i in range(1, x):
            dp[i][0] = 1
            for j in range(1, i + 1):
                # comb(i, j) = comb(i-1, j-1) + comb(i-1, j)
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % k

        # prefix_sumcounter
        cnt = [[0] * (x + 1) for _ in range(x)]
        for i in range(x):
            for j in range(i + 1):
                cnt[i][j + 1] = cnt[i][j] + int(dp[i][j] % k == 0)

        # 查询
        for _ in range(t):
            n, m = ac.read_list_ints()
            ans = 0
            for i in range(0, n + 1):
                ans += cnt[i][min(i + 1, m + 1)]
            ac.st(ans)
        return

    @staticmethod
    def lg_p3223(ac=FastIO()):
        # inclusion_exclusion与partition_method
        n, m = ac.read_list_ints()
        ans1 = math.factorial(n + 2) * math.factorial(m) * math.comb(n + 3, m)
        ans2 = math.factorial(2) * math.factorial(n + 1) * math.factorial(m) * math.comb(n + 2, m)
        ac.st(ans1 - ans2)
        return

    @staticmethod
    def lg_p3904(ac=FastIO()):
        # 递推second_stirling_number
        n, m = ac.read_list_ints()
        dp = [[0] * m for _ in range(n)]
        dp[0][0] = 1
        for i in range(1, n):
            dp[i][0] = 1
            for j in range(1, m):
                dp[i][j] = dp[i - 1][j] * (j + 1) + dp[i - 1][j - 1]
        ac.st(dp[n - 1][m - 1])
        return

    @staticmethod
    def main(ac=FastIO()):
        # 选择 n 个元素刚好有 m 个fault_perm的specific_plan数
        mod = 10 ** 9 + 7
        cb = Combinatorics(10 ** 6, mod)
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            if m > n:
                ac.st(0)
                continue
            ans = cb.comb(n, m) * cb.fault[n - m]
            ans %= mod
            ac.st(ans)
        return

    @staticmethod
    def lg_p5684(ac=FastIO()):
        # inclusion_exclusion与组合counter
        n = ac.read_int()
        mod = 10 ** 9 + 7
        cb = Combinatorics(n, mod)
        s = ac.read_str()
        cnt = Counter(s)
        odd = sum(cnt[x] % 2 for x in cnt)
        if odd > 1:
            ans = 0
        else:
            # 先回文串的个数
            lst = [cnt[x] // 2 for x in cnt if cnt[x] > 1]
            ans = 1
            s = sum(lst)
            for x in lst:
                ans *= cb.comb(s, x)
                s -= x
                ans %= mod
        # 再总的排列数
        total = 1
        s = n
        mu = 1
        for x in cnt:
            total *= cb.comb(s, cnt[x])
            s -= cnt[x]
            total %= mod
            mu *= cb.factorial(cnt[x])
            mu %= mod
        # 最后乘上 perm 全排列
        ac.st((total - ans) * mu % mod)
        return

    @staticmethod
    def lg_p6057(ac=FastIO()):
        # inclusion_exclusioncounter
        n, m = ac.read_list_ints()
        degree = [0] * n
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            degree[x] += 1
            degree[y] += 1
        ans = 0
        for i in range(n):
            ans += (n - 1 - degree[i]) * degree[i]
        ans //= 2
        ac.st(n * (n - 1) * (n - 2) // 6 - ans)
        return

    @staticmethod
    def cf_414b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/414/B
        tag: min_prime|partition_method|counter|dp
        """
        mod = 10 ** 9 + 7
        n, k = ac.read_list_ints()
        mp = PrimeFactor(n)
        rp = Combinatorics(15 + 1, mod)
        cnt = [0] * (n + 1)  # 当前值 last 的最小质因数幂次
        res = [0] * (n + 1)  # 结尾为 last 的数组个数
        cnt[1] = res[1] = ans = 1
        for last in range(2, n + 1):
            p = mp.min_prime[last]
            pre = last // p
            if mp.min_prime[pre] == p:
                cnt[last] = cnt[pre] + 1
                res[last] = res[pre] * (k + cnt[last] - 1) * rp.inv(cnt[last]) % mod
            else:
                cnt[last] = 1
                res[last] = res[pre] * k % mod
            ans = (ans + res[last]) % mod
        ac.st(ans)
        return

    @staticmethod
    def lc_1735(queries: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.com/problems/count-ways-to-make-array-with-product/
        tag: prime_factorization|partition_method|classical
        """
        mod = 10 ** 9 + 7
        nt = PrimeFactor(10 ** 4)
        cb = Combinatorics(10 ** 4 + 15, mod)

        # prime_factorization|与partition_method应用
        ans = []
        for n, k in queries:
            cur = 1
            for _, c in nt.prime_factor[k]:
                cur *= cb.comb(n + c - 1, n - 1)
                cur %= mod
            ans.append(cur)
        return ans

    @staticmethod
    def lc_1866(n: int, k: int) -> int:
        """
        url: https://leetcode.com/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/
        tag: stirling_number|first_kind_stirling_number
        """
        # 第一类stirling_number
        mod = 10 ** 9 + 7
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(n):
            for j in range(k):
                dp[i + 1][j + 1] = (dp[i][j] + dp[i][j + 1] * i) % mod
        return dp[n][k]

    @staticmethod
    def abc_132d(ac=FastIO()):
        # 组合mathcounter，和为 X 的长为 Y 的正整数与非负整数方程解个数
        n, k = ac.read_list_ints()
        mod = 10 ** 9 + 7
        cb = Combinatorics(n, mod)
        for i in range(1, k + 1):
            if n - k < i - 1:
                ac.st(0)
                continue
            blue = cb.comb(k - 1, i - 1)
            if i == 1:
                red = n - k + 1
            else:
                red = 0
                for mid in range(i - 1, n - k + 1):
                    red += cb.comb(mid - 1, i - 2) * (n - k - mid + 1)
                    red %= mod
            ans = blue * red
            ac.st(ans % mod)

        return

    @staticmethod
    def ac_4002(ac=FastIO()):
        # matrix_dp转化为partition_method组合数求解
        m, n = ac.read_list_ints()
        cb = Combinatorics(2 * n + m, 10 ** 9 + 7)
        ac.st(cb.comb(2 * n + m - 1, m - 1))
        return

    @staticmethod
    def ac_4496(ac=FastIO()):
        # partition_methodcounter
        mod = 998244353
        n, m, k = ac.read_list_ints()
        cb = Combinatorics(n, mod)
        ans = cb.comb(n - 1, k) * pow(m - 1, k, mod) * m
        ac.st(ans % mod)
        return

    @staticmethod
    def ac_5055(ac=FastIO()):
        # 组合mathmod|求解
        mod = 10 ** 9 + 7
        n, m, k = ac.read_list_ints()
        cb = Combinatorics(m + n, mod)
        ac.st(cb.comb(n - 1, 2 * k) * cb.comb(m - 1, 2 * k) % mod)
        return