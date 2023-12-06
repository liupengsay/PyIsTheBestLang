"""
Algorithm：快速幂、矩阵快速幂DP、乘法逆元
Function：高效整数的幂次方取模

====================================LeetCode====================================
450（https://leetcode.com/problems/number-of-distinct-binary-strings-after-applying-operations/）brain_teaser快速幂
1931（https://leetcode.com/problems/painting-a-grid-with-three-different-colors/）转移DP可以快速幂
8020（https://leetcode.com/problems/string-transformation/description/）KMP与快速幂转移
1622（https://leetcode.com/problems/fancy-sequence/description/）reverse_thinking，乘法逆元运用，类似inclusion_exclusion

=====================================LuoGu======================================
1630（https://www.luogu.com.cn/problem/P1630）快速幂，利用同模counter|和
1939（https://www.luogu.com.cn/problem/P1939）矩阵快速幂递推求解
1962（https://www.luogu.com.cn/problem/P1962）矩阵快速幂递推求解
3390（https://www.luogu.com.cn/problem/P3390）矩阵快速幂
3811（https://www.luogu.com.cn/problem/P3811）乘法逆元模板题
5775（https://www.luogu.com.cn/problem/P5775）从背包implemention、prefix_sum优化、到数列变换矩阵快速幂再到纯implemention
6045（https://www.luogu.com.cn/problem/P6045）brain_teaser组合counter与快速幂brute_force
6075（https://www.luogu.com.cn/problem/P6075）组合counter后快速幂
6392（https://www.luogu.com.cn/problem/P6392）公式拆解变换后快速幂
1045（https://www.luogu.com.cn/problem/P1045）位数公式转换与快速幂
3509（https://www.luogu.com.cn/problem/P3509）two_pointerimplemention寻找第k远的距离，快速幂原理跳转
1349（https://www.luogu.com.cn/problem/P1349）矩阵快速幂
2233（https://www.luogu.com.cn/problem/P2233）矩阵快速幂
2613（https://www.luogu.com.cn/problem/P2613）乘法逆元
3758（https://www.luogu.com.cn/problem/P3758）矩阵 DP 快速幂优化
5789（https://www.luogu.com.cn/problem/P5789）矩阵 DP 快速幂优化
5343（https://www.luogu.com.cn/problem/P5343）线性 DP 矩阵幂|速
8557（https://www.luogu.com.cn/problem/P8557）brain_teaser快速幂counter
8624（https://www.luogu.com.cn/problem/P8624）矩阵 DP 与快速幂

=====================================AcWing=====================================
27（https://www.acwing.com/problem/content/26/）浮点数快速幂



"""
import math

from src.mathmatics.fast_power.template import MatrixFastPower, FastPower
from src.strings.kmp.template import KMP
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_8020(s: str, t: str, k: int) -> int:
        # KMP与快速幂转移
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
        # 位数与快速幂保留后几百位数字
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
        # 利用取模分组counter与快速幂 1**b+2**b+..+a**b % mod 的值
        mod = 10 ** 4
        for _ in range(ac.read_int()):
            a, b = ac.read_list_ints()
            rest = [0] + [pow(i, b, mod) for i in range(1, mod)]
            ans = sum(rest) * (a // mod) + sum(rest[:a % mod + 1])
            ac.st(ans % mod)
        return

    @staticmethod
    def lg_p1939(ac=FastIO()):
        # 利用转移矩阵乘法公式和快速幂值
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

        # two_pointerimplemention寻找第k远的距离，快速幂原理跳转
        n, k, m = ac.read_list_ints()
        nums = ac.read_list_ints()

        ans = list(range(n))

        # two_pointer找出下一跳
        nex = [0] * n
        head = 0
        tail = k
        for i in range(n):
            while tail + 1 < n and nums[tail + 1] - \
                    nums[i] < nums[i] - nums[head]:
                head += 1
                tail += 1
            if nums[tail] - nums[i] <= nums[i] - nums[head]:
                nex[i] = head
            else:
                nex[i] = tail
        # 快速幂倍增
        while m:
            if m & 1:
                ans = [nex[ans[i]] for i in range(n)]
            nex = [nex[nex[i]] for i in range(n)]
            m >>= 1
        ac.lst([a + 1 for a in ans])
        return

    @staticmethod
    def lg_p1349(ac=FastIO()):
        # 矩阵快速幂
        p, q, a1, a2, n, m = ac.read_list_ints()
        if n == 1:
            ac.st(a1 % m)
            return
        if n == 2:
            ac.st(a2 % m)
            return
        # 建立快速幂矩阵
        mat = [[p, q], [1, 0]]
        res = MatrixFastPower().matrix_pow(mat, n - 2, m)
        # 结果
        ans = res[0][0] * a2 + res[0][1] * a1
        ans %= m
        ac.st(ans)
        return

    @staticmethod
    def lg_p2233(ac=FastIO()):
        # 矩阵快速幂
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
        # 乘法逆元求解
        mod = 19260817
        a = ac.read_int()
        b = ac.read_int()
        ans = a * pow(b, -1, mod)
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p3758(ac=FastIO()):
        # 矩阵 DP 快速幂优化
        n, m = ac.read_list_ints()
        # 转移矩阵
        grid = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            grid[i][i] = 1
            grid[0][i] = 1
        for _ in range(m):
            u, v = ac.read_list_ints()
            grid[u][v] = grid[v][u] = 1
        # 快速幂与最终状态
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
        # 线性 DP 矩阵幂|速
        mod = 10 ** 9 + 7
        n = ac.read_int()
        ac.read_int()
        pre = set(ac.read_list_ints())
        ac.read_int()
        pre = sorted(list(pre.intersection(set(ac.read_list_ints()))))
        size = max(pre)
        dp = [0] * (size + 1)
        dp[0] = 1
        for i in range(1, size + 1):
            for j in pre:
                if i < j:
                    break
                dp[i] += dp[i - j]
            dp[i] %= mod
        if n <= size:
            ac.st(dp[n])
            return
        # 矩阵幂|速
        mat = [[0] * (size + 1) for _ in range(size + 1)]
        for i in range(size, 0, -1):
            mat[i][-(size - i + 2)] = 1
        for j in pre:
            mat[0][j - 1] = 1
        res = MatrixFastPower().matrix_pow(mat, n - size, mod)
        ans = 0
        for j in range(size + 1):
            ans += res[0][j] * dp[size - j]
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p8557(ac=FastIO()):
        # brain_teaser快速幂counter
        mod = 998244353
        n, k = ac.read_list_ints()
        ans = pow((pow(2, k, mod) - 1) % mod, n, mod)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8624(ac=FastIO()):
        # 矩阵 DP 与快速幂
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
        # 浮点数快速幂
        if base == 0:
            return 0
        if exponent == 0:
            return 1
        return FastPower().float_fast_pow(base, exponent)