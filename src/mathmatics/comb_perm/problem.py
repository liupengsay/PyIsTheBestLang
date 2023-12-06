"""

Algorithm：数学排列组合计数、乘法逆元（也叫combinatorics）、Lucas定理
Function：全排列计数，选取comb计数，隔板法，错位排列，斯特林数、卡特兰数，容斥原理，可以通过乘法逆元快速求解组合数与全排列数
Lucas定理（comb(n, m)%p = comb(n%p, m%p)*comb(n//p, m//p)）%p

====================================LeetCode====================================
96（https://leetcode.com/problems/unique-binary-search-trees/）经典卡特兰数
95（https://leetcode.com/problems/unique-binary-search-trees/）经典卡特兰数思想进行递归，生成具体方案
634（https://leetcode.com/problems/find-the-derangement-of-an-array/）错位排列计数使用动态规划转移计算
1259（https://leetcode.com/problems/handshakes-that-dont-cross/）经典卡特兰数
2338（https://leetcode.com/problems/count-the-number-of-ideal-arrays/）使用隔板法与因数分解进行组合方案数求解
1735（https://leetcode.com/problems/count-ways-to-make-array-with-product/）经典质数分解与隔板法应用
1621（https://leetcode.com/problems/number-of-sets-of-k-non-overlapping-line-segments/）类似隔板法的思想
1866（https://leetcode.com/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/）第一类斯特林数
1916（https://leetcode.com/problems/count-ways-to-build-rooms-in-an-ant-colony/）树形DP加组合数学计数
D - Blue and Red Balls（https://atcoder.jp/contests/abc132/tasks/abc132_d）组合数学经典计数，和为 X 的长为 Y 的正整数与非负整数方程解个数

=====================================LuoGu======================================
4071（https://www.luogu.com.cn/problem/P4071）通过乘法逆元快速求解组合数与全排列数，同时递归计算错位排列数
1287（https://www.luogu.com.cn/problem/P1287）第二类斯特林数形式的DP，以及全排列数
1375（https://www.luogu.com.cn/problem/P1375）卡特兰数
1754（https://www.luogu.com.cn/problem/P1754）卡特兰数
2193（https://www.luogu.com.cn/problem/P2193）使用隔板法与因数分解进行组合方案数求解
1338（https://www.luogu.com.cn/problem/P1338）枚举满足个数的逆序对排列，即找特定逆序对个数的最小排列
1313（https://www.luogu.com.cn/problem/P1313）二项式展开的系数计算
1061（https://www.luogu.com.cn/problem/P1061）模拟计算下一个字典序排列
3197（https://www.luogu.com.cn/problem/P3197）计数快速幂计算加容斥原理
3414（https://www.luogu.com.cn/problem/P3414）组合数奇偶对半开，快速幂计算
4369（https://www.luogu.com.cn/problem/P4369）脑筋急转弯进行组合数加和构造
5520（https://www.luogu.com.cn/problem/P5520）隔板法计算组合数
3807（https://www.luogu.com.cn/problem/P3807）卢卡斯模板题
1044（https://www.luogu.com.cn/problem/P1044）卡特兰数
1655（https://www.luogu.com.cn/problem/P1655）矩阵DP，斯特林数
1680（https://www.luogu.com.cn/problem/P1680）隔板法计算不同分组的个数，使用乘法逆元与Lucas定理快速计算Comb(a,b)%m
2265（https://www.luogu.com.cn/problem/P2265）排列组合，计算comb(n+m, m)
2638（https://www.luogu.com.cn/problem/P2638）隔板法 a 个球放入 n 个盒子不要求每个都放也不要求放完的方案数
2822（https://www.luogu.com.cn/problem/P2822）组合数 comb(i, j) % k == 0 的个数计算
3223（https://www.luogu.com.cn/problem/P3223）使用容斥原理和隔板法计算
3904（https://www.luogu.com.cn/problem/P3904）递推第二类斯特林数
4071（https://www.luogu.com.cn/problem/P4071）经典错排选择 n 个元素刚好有 m 个错位排列的方案数
5684（https://www.luogu.com.cn/problem/P5684）容斥原理与组合计数
6057（https://www.luogu.com.cn/problem/P6057）容斥原理计数

===================================CodeForces===================================
1795D（https://codeforces.com/problemset/problem/1795/D）组合计数取模与乘法逆元快速计算
300C（https://codeforces.com/problemset/problem/300/C）枚举个数并使用组合数计算方案数
559C（https://codeforces.com/problemset/problem/559/C）容斥原理组合计数
1436C（https://codeforces.com/problemset/problem/1436/C）二分加组合数计算
414B（https://codeforces.com/problemset/problem/414/B）经典使用最小质因数与隔板法计数 DP
1879C（https://codeforces.com/contest/1879/problem/C）贪心枚举与组合计数


====================================AtCoder=====================================
D - Iroha and a Grid（https://atcoder.jp/contests/abc042/tasks/arc058_b）容斥原理组合计数
D - 11（https://atcoder.jp/contests/abc066/tasks/arc077_b）经典容斥原理组合计数
D - Factorization（https://atcoder.jp/contests/abc110/tasks/abc110_d）质因数分解与隔板法计数
E - Cell Distance（https://atcoder.jp/contests/abc127/tasks/abc127_e）经典贡献法组合计数

=====================================AcWing=====================================
130（https://www.acwing.com/problem/content/132/）超大数字的卡特兰数计算
4002（https://www.acwing.com/problem/content/4005/）经典矩阵DP转换为隔板法计算求解
4496（https://www.acwing.com/problem/content/4499/）经典隔板法计数
5055（https://www.acwing.com/problem/content/5058/）经典组合数学取模求解


卡特兰数（https://oi-wiki.org/math/combinatorics/catalan/）
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
        # 模板：质因数分解与隔板法计数
        n, m = ac.read_list_ints()
        mod = 10 ** 9 + 7
        cb = Combinatorics(n + 100, mod)  # 注意这里会超出n
        ans = 1
        for _, c in NumberTheory().get_prime_factor(m):
            ans *= cb.comb(c + n - 1, n - 1)  # 经典n个正整数和为c+n转换
            # 等价于sum(cb.comb(n, k)*cb.comb(c-1, k-1) for k in range(1, c+1))
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def cf_1436c(ac=FastIO()):

        # 模板：二分查找加组合数计算
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

        # 模板：容斥原理组合数计算
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
        # 模板：卡特兰数计算
        n = num_people // 2
        if num_people <= 1:
            return 1
        mod = 10 ** 9 + 7
        cm = Combinatorics(2 * n + 2, mod)
        ans = cm.comb(2 * n, n) - cm.comb(2 * n, n - 1)
        return ans % mod

    @staticmethod
    def lc_1259_2(num_people: int) -> int:
        # 模板：卡特兰数的数组形式
        n = num_people // 2
        mod = 10 ** 9 + 7
        dp = [1] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = sum(dp[j] * dp[i - 1 - j] for j in range(i)) % mod
        return dp[n]

    @staticmethod
    def lg_p1375(ac=FastIO()):
        # 模板：卡特兰数计算
        n = ac.read_int()
        mod = 10 ** 9 + 7
        cm = Combinatorics(2 * n + 2, mod)
        ans = cm.comb(2 * n, n) - cm.comb(2 * n, n - 1)
        ac.st(ans % mod)
        return

    @staticmethod
    def lg_p1754(ac=FastIO()):
        # 模板：卡特兰数计算
        n = ac.read_int()
        # 卡特兰数的另一种区间递推形式 dp[i][j] = dp[i-1][j]+dp[i][j-1]
        # 类似题目也有长为 2n 合法的括号匹配数 h(n) = h(n-1)*(4*n-2)//(n+1)
        # 也可以使用 h(n) = math.comb(2*n, n)//(n+1) 求解
        ans = math.comb(2 * n, n) - math.comb(2 * n, n - 1)
        assert ans == math.comb(2 * n, n) // (n + 1)  # 不需要取模时可以直接用这个计算
        ac.st(ans)
        return

    @staticmethod
    def lc_634(n):
        # 模板：求错位排列组合数
        mod = 10 ** 9 + 7
        fault = [0, 0, 1, 2]
        for i in range(4, n + 1):
            fault.append((i - 1) * (fault[i - 1] + fault[i - 2]) % mod)
        return fault[n]

    @staticmethod
    def cf_300c(ac=FastIO()):
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
        # 模板：组合数与错位排列求解
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
            # 斯特林数，把a个球放入b个盒子且不允许空盒的方案数
            if a < b or b < 0:
                return 0
            if a == b:
                return 1
            # 新球单独放新盒子，或者放入已经有的老盒子
            return dfs(a - 1, b - 1) + b * dfs(a - 1, b)

        # 由于是不同的球还要计算球的全排列
        x = dfs(n, r) * math.factorial(r)
        return x

    @staticmethod
    def lg_p1287(ac=FastIO()):
        # 模板：容斥原理计数
        n, r = ac.read_list_ints()
        ans = 0
        for k in range(r):
            cur = ((-1) ** k) * math.comb(r, k) * ((r - k) ** n)
            ans += cur
        ac.st(ans)
        return

    @staticmethod
    def lg_p4071(ac=FastIO()):
        # 模板：隔板法计算组合数
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
        # 模板：Lucas模板题
        for _ in range(ac.read_int()):
            n, m, p = ac.read_list_ints()
            ans = Lucas().lucas_iter(n + m, n, p)
            ac.st(ans)
        return

    @staticmethod
    def abc_42d(ac=FastIO()):
        # 模板：容斥原理组合计数
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
        # 模板：经典容斥原理组合计数
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
        # 模板：经典贡献法组合计数
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
        # 模板：超大范围的卡特兰数计算 h(n) = C(2n, n)//(n+1) = ((n+1)*..*(2*n))//(1*2*..*(n+1))
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
        # 模板：第二类斯特林数只能递推（n个不同的球放入m个相同的盒子，不允许为空，使用斯特林数计算）
        n = m = 101  # （n个相同的球放入m个不同的盒子，不允许为空，使用隔板法计算）
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
        # 模板：隔板法计算不同分组的个数，使用乘法逆元与Lucas定理快速计算Comb(a,b) % m
        # 经典转换为（n个相同的球放入m个不同的盒子，不允许为空的方案数）
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
        # 模板：排列组合，计算comb(n+m, m)
        mod = 1000000007
        n, m = ac.read_list_ints()
        ans = Lucas().comb(n + m, m, mod)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2638(ac=FastIO()):
        # 模板：隔板法 a 个球放入 n 个盒子不要求每个都放也不要求放完的方案数
        n, a, b = ac.read_list_ints()
        ans = math.comb(n + a, n) * math.comb(n + b, n)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2822(ac=FastIO()):
        # 模板：组合数 comb(i, j) % k == 0 的个数计算
        t, k = ac.read_list_ints()

        # 预处理计算组合数  comb(i, j) % k
        x = 2001
        dp = [[0] * x for _ in range(x)]
        dp[0][0] = 1
        for i in range(1, x):
            dp[i][0] = 1
            for j in range(1, i + 1):
                # comb(i, j) = comb(i-1, j-1) + comb(i-1, j)
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % k

        # 前缀和计数
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
        # 模板：使用容斥原理与隔板法计算
        n, m = ac.read_list_ints()
        ans1 = math.factorial(n + 2) * math.factorial(m) * math.comb(n + 3, m)
        ans2 = math.factorial(2) * math.factorial(n + 1) * math.factorial(m) * math.comb(n + 2, m)
        ac.st(ans1 - ans2)
        return

    @staticmethod
    def lg_p3904(ac=FastIO()):
        # 模板：递推第二类斯特林数
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
        # 模板：选择 n 个元素刚好有 m 个错位排列的方案数
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
        # 模板：容斥原理与组合计数
        n = ac.read_int()
        mod = 10 ** 9 + 7
        cb = Combinatorics(n, mod)
        s = ac.read_str()
        cnt = Counter(s)
        odd = sum(cnt[x] % 2 for x in cnt)
        if odd > 1:
            ans = 0
        else:
            # 先计算回文串的个数
            lst = [cnt[x] // 2 for x in cnt if cnt[x] > 1]
            ans = 1
            s = sum(lst)
            for x in lst:
                ans *= cb.comb(s, x)
                s -= x
                ans %= mod
        # 再计算总的排列数
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
        # 模板：容斥原理计数
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
        mod = 10 ** 9 + 7
        nt = PrimeFactor(10 ** 4)
        cb = Combinatorics(10 ** 4 + 15, mod)

        # 模板：经典质数分解与隔板法应用
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
        # 模板：第一类斯特林数
        mod = 10 ** 9 + 7
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(n):
            for j in range(k):
                dp[i + 1][j + 1] = (dp[i][j] + dp[i][j + 1] * i) % mod
        return dp[n][k]

    @staticmethod
    def abc_132d(ac=FastIO()):
        # 模板：组合数学经典计数，和为 X 的长为 Y 的正整数与非负整数方程解个数
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
        # 模板：矩阵DP转化为隔板法组合数求解
        m, n = ac.read_list_ints()
        cb = Combinatorics(2 * n + m, 10 ** 9 + 7)
        ac.st(cb.comb(2 * n + m - 1, m - 1))
        return

    @staticmethod
    def ac_4496(ac=FastIO()):
        # 模板：经典隔板法计数
        mod = 998244353
        n, m, k = ac.read_list_ints()
        cb = Combinatorics(n, mod)
        ans = cb.comb(n - 1, k) * pow(m - 1, k, mod) * m
        ac.st(ans % mod)
        return

    @staticmethod
    def ac_5055(ac=FastIO()):
        # 模板：经典组合数学取模求解
        mod = 10 ** 9 + 7
        n, m, k = ac.read_list_ints()
        cb = Combinatorics(m + n, mod)
        ac.st(cb.comb(n - 1, 2 * k) * cb.comb(m - 1, 2 * k) % mod)
        return