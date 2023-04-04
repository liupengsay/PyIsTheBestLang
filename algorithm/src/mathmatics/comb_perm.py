from itertools import combinations
from itertools import permutations
import unittest
from algorithm.src.fast_io import FastIO
from typing import List
from collections import Counter
from algorithm.src.fast_io import FastIO
import math
from functools import lru_cache

"""

算法：数学排列组合计数、乘法逆元（也叫combinatorics）
功能：全排列计数，选取comb计数，隔板法，错位排列，斯特林数、卡特兰数，容斥原理，可以通过乘法逆元快速求解组合数与全排列数
题目：

===================================力扣===================================
634. 寻找数组的错位排列（https://leetcode.cn/problems/find-the-derangement-of-an-array/）错位排列计数使用动态规划转移计算
1259. 不相交的握手（https://leetcode.cn/problems/handshakes-that-dont-cross/）卡特兰数
2338. 统计理想数组的数目（https://leetcode.cn/problems/count-the-number-of-ideal-arrays/）使用隔板法与因数分解进行组合方案数求解

===================================洛谷===================================
P4071 排列计数（https://www.luogu.com.cn/problem/P4071）通过乘法逆元快速求解组合数与全排列数，同时递归计算错位排列数
P1287 盒子与球（https://www.luogu.com.cn/problem/P1287）斯特林数形式的DP，以及全排列数
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

================================CodeForces================================
D. Triangle Coloring（https://codeforces.com/problemset/problem/1795/D）组合计数取模与乘法逆元快速计算
C. Beautiful Numbers（https://codeforces.com/problemset/problem/300/C）枚举个数并使用组合数计算方案数
C. Gerald and Giant Chess（https://codeforces.com/problemset/problem/559/C）容斥原理组合计数
C. Binary Search（https://codeforces.com/problemset/problem/1436/C）二分加组合数计算

参考：OI WiKi（xx）
卡特兰数（https://oi-wiki.org/math/combinatorics/catalan/）
"""


class Combinatorics:
    def __init__(self, n, mod):
        # 模板：求全排列组合数
        self.perm = [1] * n
        self.rev = [1] * n
        self.mod = mod
        for i in range(1, n):
            # 阶乘数 i! 取模
            self.perm[i] = self.perm[i - 1] * i
            self.perm[i] %= self.mod
        self.rev[-1] = pow(self.perm[-1], -1, self.mod)
        for i in range(n-2, 0, -1):
            self.rev[i] = (self.rev[i+1]*(i+1)%mod)
        return

    def comb(self, a, b):
        # 组合数根据乘法逆元求解
        res = self.perm[a] * self.rev[b] * self.rev[a - b]
        return res % self.mod

    def factorial(self, a):
        # 组合数根据乘法逆元求解
        res = self.perm[a]
        return res % self.mod


class Solution:
    def __int__(self):
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
    def main_p4071():
        import sys
        sys.setrecursionlimit(10 ** 8)

        def read():
            return sys.stdin.readline().strip()

        def ac(x):
            return sys.stdout.write(str(x) + '\n')

        length = 100
        mod = 10 ** 9 + 7

        # 求全排列组合数
        perm = [1] * length
        for i in range(1, length):
            perm[i] = perm[i - 1] * i
            perm[i] %= mod

        # 求错位排列组合数
        fault = [0] * length
        fault[0] = 1
        fault[2] = 1
        for i in range(3, length):
            fault[i] = (i - 1) * (fault[i - 1] + fault[i - 2])
            fault[i] %= mod

        # 利用乘法逆元求解组合数
        def comb(a, b):
            res = perm[a] * pow(perm[b], -1, mod) * pow(perm[a - b], -1, mod)
            return res % mod

        def main():
            t = int(input())
            for _ in range(t):
                n, m = [int(w) for w in input().strip().split() if w]
                while len(fault) <= n:
                    num = (len(fault) - 1) * (fault[-1] + fault[-2])
                    num %= mod
                    fault.append(num)

                    num = len(perm) * perm[-1]
                    num %= mod
                    perm.append(num)

                if m > n:
                    ac(0)
                else:
                    # m 个全排列乘 n-m 个错位排列
                    ans = (comb(n, m) * fault[n - m]) % mod
                    ac(ans)
            return

        main()
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


class TestGeneral(unittest.TestCase):

    def test_comb_perm(self):
        cp = Solution()
        i = 500
        j = 10000
        mod = 10**9 + 7
        assert math.comb(j, i) % mod == cp.comb_perm(j, i)

        assert cp.main_p1287(3, 2) == 6

        nums = [1, 2, 3]
        ans = cp.combinnation(nums, 2)
        assert ans == [[1, 2], [1, 3], [2, 3]]

        ans = cp.permutation(nums, 1)
        assert ans == [[1], [2], [3]]

        return


if __name__ == '__main__':
    unittest.main()
