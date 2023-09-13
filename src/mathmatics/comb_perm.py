import unittest
from typing import List
from collections import Counter, defaultdict
from src.fast_io import FastIO
import math
from functools import lru_cache

from src.mathmatics.number_theory import NumberTheoryPrimeFactor, NumberTheory

"""

算法：数学排列组合计数、乘法逆元（也叫combinatorics）、Lucas定理
功能：全排列计数，选取comb计数，隔板法，错位排列，斯特林数、卡特兰数，容斥原理，可以通过乘法逆元快速求解组合数与全排列数
题目：
Lucas定理（comb(n, m)%p = comb(n%p, m%p)*comb(n//p, m//p)）%p

===================================力扣===================================
96. 不同的二叉搜索树（https://leetcode.cn/problems/unique-binary-search-trees/）经典卡特兰数
95. 不同的二叉搜索树 II（https://leetcode.cn/problems/unique-binary-search-trees/）经典卡特兰数思想进行递归，生成具体方案
634. 寻找数组的错位排列（https://leetcode.cn/problems/find-the-derangement-of-an-array/）错位排列计数使用动态规划转移计算
1259. 不相交的握手（https://leetcode.cn/problems/handshakes-that-dont-cross/）经典卡特兰数
2338. 统计理想数组的数目（https://leetcode.cn/problems/count-the-number-of-ideal-arrays/）使用隔板法与因数分解进行组合方案数求解
1735. 生成乘积数组的方案数（https://leetcode.cn/problems/count-ways-to-make-array-with-product/）经典质数分解与隔板法应用
1621. 大小为 K 的不重叠线段的数目（https://leetcode.cn/problems/number-of-sets-of-k-non-overlapping-line-segments/）类似隔板法的思想
1866. 恰有 K 根木棍可以看到的排列数目（https://leetcode.cn/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/）第一类斯特林数
1916. 统计为蚁群构筑房间的不同顺序（https://leetcode.cn/problems/count-ways-to-build-rooms-in-an-ant-colony/）树形DP加组合数学计数

===================================洛谷===================================
P4071 排列计数（https://www.luogu.com.cn/problem/P4071）通过乘法逆元快速求解组合数与全排列数，同时递归计算错位排列数
P1287 盒子与球（https://www.luogu.com.cn/problem/P1287）第二类斯特林数形式的DP，以及全排列数
P1375 小猫（https://www.luogu.com.cn/problem/P1375）卡特兰数
P1754 球迷购票问题（https://www.luogu.com.cn/problem/P1754）卡特兰数
P2193 HXY和序列（https://www.luogu.com.cn/problem/P2193）使用隔板法与因数分解进行组合方案数求解
P1338 末日的传说（https://www.luogu.com.cn/problem/P1338）枚举满足个数的逆序对排列，即找特定逆序对个数的最小排列
P1313 [NOIP2011 提高组] 计算系数（https://www.luogu.com.cn/problem/P1313）二项式展开的系数计算
P1061 [NOIP2006 普及组] Jam 的计数法（https://www.luogu.com.cn/problem/P1061）模拟计算下一个字典序排列
P3197 [HNOI2008]越狱（https://www.luogu.com.cn/problem/P3197）计数快速幂计算加容斥原理
P3414 SAC#1 - 组合数（https://www.luogu.com.cn/problem/P3414）组合数奇偶对半开，快速幂计算
P4369 [Code+#4]组合数问题（https://www.luogu.com.cn/problem/P4369）脑筋急转弯进行组合数加和构造
P5520 [yLOI2019] 青原樱（https://www.luogu.com.cn/problem/P5520）隔板法计算组合数
P3807 【模板】卢卡斯定理/Lucas 定理（https://www.luogu.com.cn/problem/P3807）卢卡斯模板题
P1044 [NOIP2003 普及组] 栈（https://www.luogu.com.cn/problem/P1044）卡特兰数
P1655 小朋友的球（https://www.luogu.com.cn/problem/P1655）矩阵DP，斯特林数
P1680 奇怪的分组（https://www.luogu.com.cn/problem/P1680）隔板法计算不同分组的个数，使用乘法逆元与Lucas定理快速计算Comb(a,b)%m
P2265 路边的水沟（https://www.luogu.com.cn/problem/P2265）排列组合，计算comb(n+m, m)
P2638 安全系统（https://www.luogu.com.cn/problem/P2638）隔板法 a 个球放入 n 个盒子不要求每个都放也不要求放完的方案数
P2822 [NOIP2016 提高组] 组合数问题（https://www.luogu.com.cn/problem/P2822）组合数 comb(i, j) % k == 0 的个数计算 
P3223 [HNOI2012] 排队（https://www.luogu.com.cn/problem/P3223）使用容斥原理和隔板法计算
P3904 三只小猪（https://www.luogu.com.cn/problem/P3904）递推第二类斯特林数
P4071 [SDOI2016]排列计数（https://www.luogu.com.cn/problem/P4071）经典错排选择 n 个元素刚好有 m 个错位排列的方案数
P5684 [CSP-J2019 江西] 非回文串（https://www.luogu.com.cn/problem/P5684）容斥原理与组合计数
P6057 [加油武汉]七步洗手法（https://www.luogu.com.cn/problem/P6057）容斥原理计数

================================CodeForces================================
D. Triangle Coloring（https://codeforces.com/problemset/problem/1795/D）组合计数取模与乘法逆元快速计算
C. Beautiful Numbers（https://codeforces.com/problemset/problem/300/C）枚举个数并使用组合数计算方案数
C. Gerald and Giant Chess（https://codeforces.com/problemset/problem/559/C）容斥原理组合计数
C. Binary Search（https://codeforces.com/problemset/problem/1436/C）二分加组合数计算
B. Mashmokh and ACM（https://codeforces.com/problemset/problem/414/B）经典使用最小质因数与隔板法计数 DP


================================AtCoder================================
D - Iroha and a Grid（https://atcoder.jp/contests/abc042/tasks/arc058_b）容斥原理组合计数
D - 11（https://atcoder.jp/contests/abc066/tasks/arc077_b）经典容斥原理组合计数
D - Factorization（https://atcoder.jp/contests/abc110/tasks/abc110_d）质因数分解与隔板法计数

================================AcWing==================================
130. 火车进出栈问题（https://www.acwing.com/problem/content/132/）超大数字的卡特兰数计算
4002. 构造数组（https://www.acwing.com/problem/content/4005/）经典矩阵DP转换为隔板法计算求解
4496. 吃水果（https://www.acwing.com/problem/content/4499/）经典隔板法计数
5055. 画矩形（https://www.acwing.com/problem/content/5058/）经典组合数学取模求解


参考：OI WiKi（xx）
卡特兰数（https://oi-wiki.org/math/combinatorics/catalan/）
"""


class Combinatorics:
    def __init__(self, n, mod):
        # 模板：求全排列组合数，使用时注意 n 的取值范围
        n += 10
        self.perm = [1] * n
        self.rev = [1] * n
        self.mod = mod
        for i in range(1, n):
            # 阶乘数 i! 取模
            self.perm[i] = self.perm[i - 1] * i
            self.perm[i] %= self.mod
        self.rev[-1] = self.mod_reverse(self.perm[-1], self.mod)  # 等价于pow(self.perm[-1], -1, self.mod)
        for i in range(n - 2, 0, -1):
            self.rev[i] = (self.rev[i + 1] * (i + 1) % mod)  # 阶乘 i! 取逆元
        self.fault = [0] * n
        self.fault_perm()
        return

    def ex_gcd(self, a, b):
        # 扩展欧几里得求乘法逆元
        if b == 0:
            return 1, 0, a
        else:
            x, y, q = self.ex_gcd(b, a % b)
            x, y = y, (x - (a // b) * y)
            return x, y, q

    def mod_reverse(self, a, p):
        x, y, q = self.ex_gcd(a, p)
        if q != 1:
            raise Exception("No solution.")   # 逆元要求a与p互质
        else:
            return (x + p) % p  # 防止负数

    def comb(self, a, b):
        if a < b:
            return 0
        # 组合数根据乘法逆元求解
        res = self.perm[a] * self.rev[b] * self.rev[a - b]
        return res % self.mod

    def factorial(self, a):
        # 组合数根据乘法逆元求解
        res = self.perm[a]
        return res % self.mod

    def fault_perm(self):
        # 求错位排列组合数
        self.fault[0] = 1
        self.fault[2] = 1
        for i in range(3, len(self.fault)):
            self.fault[i] = (i - 1) * (self.fault[i - 1] + self.fault[i - 2])
            self.fault[i] %= self.mod
        return

    def inv(self, n):
        # 求 pow(n, -1, mod)
        return self.perm[n - 1] * self.rev[n] % self.mod

    def catalan(self, n):
        # 求卡特兰数
        return (self.comb(2 * n, n) - self.comb(2 * n, n - 1)) % self.mod


class Lucas:
    def __init__(self):
        # 模板：快速求Comb(a,b)%p
        return

    @staticmethod
    def lucas(self, n, m, p):
        # 模板：卢卡斯定理，求 math.comb(n, m) % p，要求p为质数
        if m == 0:
            return 1
        return ((math.comb(n % p, m % p) % p) * self.lucas(n // p, m // p, p)) % p

    @staticmethod
    def comb(n, m, p):
        # 模板：利用乘法逆元求comb(n,m)%p
        ans = 1
        for x in range(n - m + 1, n + 1):
            ans *= x
            ans %= p
        for x in range(1, m + 1):
            ans *= pow(x, -1, p)
            ans %= p
        return ans

    def lucas_iter(self, n, m, p):
        # 模板：卢卡斯定理，求 math.comb(n, m) % p，要求p为质数
        if m == 0:
            return 1
        stack = [[n, m]]
        dct = dict()
        while stack:
            n, m = stack.pop()
            if n >= 0:
                if m == 0:
                    dct[(n, m)] = 1
                    continue
                stack.append([~n, m])
                stack.append([n // p, m // p])
            else:
                n = ~n
                dct[(n, m)] = (self.comb(n % p, m % p, p) % p) * dct[(n // p, m // p)] % p
        return dct[(n, m)]

    @staticmethod
    def extend_lucas(self, n, m, p):
        # 模板：扩展卢卡斯定理，求 math.comb(n, m) % p，不要求p为质数
        return


class Solution:
    def __int__(self):
        return

    @staticmethod
    def abc_110d(ac=FastIO()):
        # 模板：质因数分解与隔板法计数
        n, m = ac.read_ints()
        mod = 10**9 + 7
        cb = Combinatorics(n+100, mod)  # 注意这里会超出n
        ans = 1
        for _, c in NumberTheory().get_prime_factor(m):
            ans *= cb.comb(c+n-1, n-1)  # 经典n个正整数和为c+n转换
            # 等价于sum(cb.comb(n, k)*cb.comb(c-1, k-1) for k in range(1, c+1))
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def cf_1436c(ac=FastIO()):

        # 模板：二分查找加组合数计算
        n, x, pos = ac.read_ints()
        big = small = 0

        left = 0
        right = n
        while left < right:
            mid = (left+right)//2
            if mid <= pos:
                small += int(mid != pos)
                left = mid+1
            else:
                right = mid
                big += 1

        if small >= x or big > n - x:
            ac.st(0)
            return
        mod = 10**9+7
        comb = Combinatorics(n, mod)
        ans = comb.comb(n-x, big)*comb.factorial(big)*math.comb(x-1, small)*math.factorial(small)
        ans *= comb.factorial(n-big-small-1)
        ac.st(ans % mod)
        return

    @staticmethod
    def cf_559c(ac=FastIO()):

        # 模板：容斥原理组合数计算
        m, n, k = ac.read_ints()
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
        assert ans == math.comb(2*n, n)//(n+1)  # 不需要取模时可以直接用这个计算
        ac.st(ans)
        return

    @staticmethod
    def lc_634(n):
        # 模板：求错位排列组合数
        mod = 10**9+7
        fault = [0, 0, 1, 2]
        for i in range(4, n + 1):
            fault.append((i - 1) * (fault[i - 1] + fault[i - 2]) % mod)
        return fault[n]

    @staticmethod
    def cf_300c(ac=FastIO()):
        mod = 10 ** 9 + 7
        a, b, n = ac.read_ints()
        c = Combinatorics(n+1, mod)

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
        mod = 10**9+7
        cb = Combinatorics(10**6, mod)
        for _ in range(ac.read_int()):
            n, m = ac.read_ints()
            if m > n:
                ac.st(0)
                continue
            ans = cb.comb(n, m)*cb.fault[n-m]
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
        n, r = ac.read_ints()
        ans = 0
        for k in range(r):
            cur = ((-1)**k)*math.comb(r, k)*((r-k)**n)
            ans += cur
        ac.st(ans)
        return

    @staticmethod
    def lg_p4071(ac=FastIO()):
        # 模板：隔板法计算组合数
        tp, n, m, p = ac.read_ints()

        if n < 2 * m - 1:
            ac.st(0)
            return

        ans = 1
        for x in range(n-2*m+2, n-m+2):
            ans *= x
            ans %= p
        ac.st(ans)
        return

    @staticmethod
    def lg_p3807(ac=FastIO()):
        # 模板：Lucas模板题
        for _ in range(ac.read_int()):
            n, m, p = ac.read_ints()
            ans = Lucas().lucas_iter(n + m, n, p)
            ac.st(ans)
        return

    @staticmethod
    def abc_42d(ac=FastIO()):
        # 模板：容斥原理组合计数
        mod = 10**9 + 7
        h, w, a, b = ac.read_ints()
        cb = Combinatorics(h+w+2, mod)
        ans = cb.comb(h+w-2, h-1)
        for x in range(h-a+1, h+1):
            y = b
            cur = cb.comb(x+y-2, x-1)*cb.comb(h-x+w-y-1, h-x)
            ans = (ans-cur) % mod
        ac.st(ans)
        return

    @staticmethod
    def abc_65d(ac=FastIO()):
        # 模板：经典容斥原理组合计数
        mod = 10**9 + 7
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
    def ac_130(ac=FastIO()):
        # 模板：超大范围的卡特兰数计算 h(n) = C(2n, n)//(n+1) = ((n+1)*..*(2*n))//(1*2*..*(n+1))
        n = ac.read_int()
        nt = NumberTheoryPrimeFactor(2*n+1)
        cnt = defaultdict(int)
        for i in range(1, 2*n+1):
            for num, y in nt.prime_factor[i]:
                if i <= n:
                    cnt[num] -= y
                else:
                    cnt[num] += y
        ans = 1
        for w in cnt:
            ans *= w**cnt[w]
        ac.st(ans // (n+1))
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
        n, m = ac.read_ints()
        n -= sum([ac.read_int() for _ in range(m)])
        m -= 1
        n -= 1
        p = 10**9+7
        ans = Lucas().comb(n, m, p)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2265(ac=FastIO()):
        # 模板：排列组合，计算comb(n+m, m)
        mod = 1000000007
        n, m = ac.read_ints()
        ans = Lucas().comb(n+m, m, mod)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2638(ac=FastIO()):
        # 模板：隔板法 a 个球放入 n 个盒子不要求每个都放也不要求放完的方案数
        n, a, b = ac.read_ints()
        ans = math.comb(n + a, n) * math.comb(n + b, n)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2822(ac=FastIO()):
        # 模板：组合数 comb(i, j) % k == 0 的个数计算
        t, k = ac.read_ints()

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
            n, m = ac.read_ints()
            ans = 0
            for i in range(0, n + 1):
                ans += cnt[i][min(i + 1, m + 1)]
            ac.st(ans)
        return

    @staticmethod
    def lg_p3223(ac=FastIO()):
        # 模板：使用容斥原理与隔板法计算
        n, m = ac.read_ints()
        ans1 = math.factorial(n + 2) * math.factorial(m) * math.comb(n + 3, m)
        ans2 = math.factorial(2) * math.factorial(n + 1) * math.factorial(m) * math.comb(n + 2, m)
        ac.st(ans1 - ans2)
        return

    @staticmethod
    def lg_p3904(ac=FastIO()):
        # 模板：递推第二类斯特林数
        n, m = ac.read_ints()
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
        mod = 10**9 + 7
        cb = Combinatorics(10**6, mod)
        for _ in range(ac.read_int()):
            n, m = ac.read_ints()
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
        mod = 10**9 + 7
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
        n, m = ac.read_ints()
        degree = [0] * n
        for _ in range(m):
            x, y = ac.read_ints_minus_one()
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
        n, k = ac.read_ints()
        mp = NumberTheoryPrimeFactor(n)
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
        nt = NumberTheoryPrimeFactor(10 ** 4)
        cb = Combinatorics(10 ** 4 + 15, mod)

        # 模板：经典质数分解与隔板法应用
        ans = []
        for n, k in queries:
            cur = 1
            for _, c in nt.prime_factor[k]:
                cur *= cb.comb(n+c-1, n-1)
                cur %= mod
            ans.append(cur)
        return ans

    @staticmethod
    def lc_1866(n: int, k: int) -> int:
        # 模板：第一类斯特林数
        mod = 10**9 + 7
        dp = [[0]*(k+1) for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(n):
            for j in range(k):
                dp[i+1][j+1] = (dp[i][j] + dp[i][j+1]*i)%mod
        return dp[n][k]

    @staticmethod
    def ac_4002(ac=FastIO()):
        # 模板：矩阵DP转化为隔板法组合数求解
        m, n = ac.read_ints()
        cb = Combinatorics(2*n+m, 10**9+7)
        ac.st(cb.comb(2*n+m-1, m-1))
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
        mod = 10**9 + 7
        n, m, k = ac.read_ints()
        cb = Combinatorics(m + n, mod)
        ac.st(cb.comb(n - 1, 2 * k) * cb.comb(m - 1, 2 * k) % mod)
        return


class TestGeneral(unittest.TestCase):
    def test_comb_perm(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
