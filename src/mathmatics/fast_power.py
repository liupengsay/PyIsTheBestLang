import copy
import math
import random
import unittest

from src.fast_io import FastIO

"""
算法：快速幂、矩阵快速幂DP、乘法逆元
功能：高效计算整数的幂次方取模
题目：

===================================力扣===================================
450. 应用操作后不同二进制字符串的数量（https://leetcode.cn/problems/number-of-distinct-binary-strings-after-applying-operations/）脑筋急转弯快速幂计算
1931. 用三种不同颜色为网格涂色（https://leetcode.cn/problems/painting-a-grid-with-three-different-colors/）转移DP可以使用快速幂进行计算

===================================洛谷===================================
P1630 求和（https://www.luogu.com.cn/problem/P1630）快速幂计算，利用同模进行计数加和
P1939 【模板】矩阵加速（数列）（https://www.luogu.com.cn/problem/P1939）矩阵快速幂递推求解
P1962 斐波那契数列（https://www.luogu.com.cn/problem/P1962）矩阵快速幂递推求解
P3390 【模板】矩阵快速幂（https://www.luogu.com.cn/problem/P3390）矩阵快速幂计算
P3811 【模板】乘法逆元（https://www.luogu.com.cn/problem/P3811）乘法逆元模板题
P5775 [AHOI2006]斐波卡契的兔子（https://www.luogu.com.cn/problem/P5775）从背包模拟、前缀和优化、到数列变换使用矩阵快速幂再到纯模拟
P5550 Chino的数列（https://www.luogu.com.cn/problem/P5550）循环节计算也可以使用矩阵快速幂递推
P6045 后缀树（https://www.luogu.com.cn/problem/P6045）脑筋急转弯进行组合计数与快速幂枚举计算
P6075 [JSOI2015]子集选取（https://www.luogu.com.cn/problem/P6075）组合计数后进行快速幂计算
P6392 中意（https://www.luogu.com.cn/problem/P6392）公式拆解变换后进行快速幂计算
P1045 [NOIP2003 普及组] 麦森数（https://www.luogu.com.cn/problem/P1045）位数公式转换与快速幂计算
P3509 [POI2010]ZAB-Frog（https://www.luogu.com.cn/problem/P3509）双指针模拟寻找第k远的距离，快速幂原理跳转
P1349 广义斐波那契数列（https://www.luogu.com.cn/problem/P1349）矩阵快速幂
P2233 [HNOI2002]公交车路线（https://www.luogu.com.cn/problem/P2233）矩阵快速幂
P2613 【模板】有理数取余（https://www.luogu.com.cn/problem/P2613）乘法逆元
P3758 [TJOI2017]可乐（https://www.luogu.com.cn/problem/P3758）矩阵 DP 使用快速幂优化
P5789 [TJOI2017]可乐（数据加强版）（https://www.luogu.com.cn/problem/P5789）矩阵 DP 使用快速幂优化
P5343 【XR-1】分块（https://www.luogu.com.cn/problem/P5343）线性 DP 使用矩阵幂加速计算
P8557 炼金术（Alchemy）（https://www.luogu.com.cn/problem/P8557）脑筋急转弯快速幂计数
P8624 [蓝桥杯 2015 省 AB] 垒骰子（https://www.luogu.com.cn/problem/P8624）矩阵 DP 与快速幂

===================================AcWing===================================
27. 数值的整数次方（https://www.acwing.com/problem/content/26/）浮点数快速幂


参考：OI WiKi（xx）

"""


class FastPower:
    def __init__(self):
        return

    @staticmethod
    def fast_power_api(a, b, mod):
        return pow(a, b, mod)

    @staticmethod
    def fast_power(a, b, mod):
        a = a % mod
        res = 1
        while b > 0:
            if b & 1:
                res = res * a % mod
            a = a * a % mod
            b >>= 1
        return res

    @staticmethod
    def float_fast_pow(x: float, n: int) -> float:
        # 浮点数快速幂
        def quick_mul(n):
            if n == 0:
                return 1.0
            y = quick_mul(n // 2)
            return y * y if n % 2 == 0 else y * y * x

        return quick_mul(n) if n >= 0 else 1.0 / quick_mul(-n)


class MatrixFastPower:
    def __init__(self):
        return

    @staticmethod
    def matrix_mul(a, b, mod=10**9 + 7):
        n = len(a)
        res = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    res[i][j] += (a[i][k] % mod) * (b[k][j] % mod)
                    res[i][j] %= mod
        return res

    def matrix_pow(self, base, p, mod=10**9 + 7):
        n = len(base)
        ans = [[0] * n for _ in range(n)]
        for i in range(n):
            ans[i][i] = 1
        while p:
            if p & 1:
                ans = self.matrix_mul(ans, base, mod)
            base = self.matrix_mul(base, base, mod)
            p >>= 1
        return ans


class PowerReverse:
    def __init__(self):
        return

    # 扩展欧几里得求乘法逆元
    def ex_gcd(self, a, b):
        if b == 0:
            return 1, 0, a
        else:
            x, y, q = self.ex_gcd(b, a % b)
            x, y = y, (x - (a // b) * y)
            return x, y, q

    def mod_reverse(self, a, p):
        x, y, q = self.ex_gcd(a, p)
        if q != 1:
            raise Exception("No solution.")
        else:
            return (x + p) % p  # 防止负数


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1045(ac=FastIO()):
        # 模板：位数计算与快速幂保留后几百位数字
        p = ac.read_int()
        ans1 = int(p * math.log10(2)) + 1
        ans2 = pow(2, p, 10**501) - 1
        ans2 = str(ans2)[-500:]
        ac.st(ans1)
        ans2 = "0" * (500 - len(ans2)) + ans2
        for i in range(0, 500, 50):
            ac.st(ans2[i:i + 50])
        return

    @staticmethod
    def lg_p1630(ac=FastIO()):
        # 模板：利用取模分组计数与快速幂计算 1**b+2**b+..+a**b % mod 的值
        mod = 10**4
        for _ in range(ac.read_int()):
            a, b = ac.read_ints()
            rest = [0] + [pow(i, b, mod) for i in range(1, mod)]
            ans = sum(rest) * (a // mod) + sum(rest[:a % mod + 1])
            ac.st(ans % mod)
        return

    @staticmethod
    def lg_p1939(ac=FastIO()):
        # 模板：利用转移矩阵乘法公式和快速幂计算值
        mat = [[1, 0, 1], [1, 0, 0], [0, 1, 0]]
        lst = [1, 1, 1]
        mod = 10**9 + 7
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

        # 模板：双指针模拟寻找第k远的距离，快速幂原理跳转
        n, k, m = ac.read_ints()
        nums = ac.read_list_ints()

        ans = list(range(n))

        # 双指针找出下一跳
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
        # 快速幂倍增计算
        while m:
            if m & 1:
                ans = [nex[ans[i]] for i in range(n)]
            nex = [nex[nex[i]] for i in range(n)]
            m >>= 1
        ac.lst([a + 1 for a in ans])
        return

    @staticmethod
    def lg_p1349(ac=FastIO()):
        # 模板：矩阵快速幂
        p, q, a1, a2, n, m = ac.read_ints()
        if n == 1:
            ac.st(a1 % m)
            return
        if n == 2:
            ac.st(a2 % m)
            return
        # 建立快速幂矩阵
        mat = [[p, q], [1, 0]]
        res = MatrixFastPower().matrix_pow(mat, n - 2, m)
        # 计算结果
        ans = res[0][0] * a2 + res[0][1] * a1
        ans %= m
        ac.st(ans)
        return

    @staticmethod
    def lg_p2233(ac=FastIO()):
        # 模板：矩阵快速幂
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
        # 模板：乘法逆元求解
        mod = 19260817
        a = ac.read_int()
        b = ac.read_int()
        ans = a * pow(b, -1, mod)
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p3758(ac=FastIO()):
        # 模板：矩阵 DP 使用快速幂优化
        n, m = ac.read_ints()
        # 转移矩阵
        grid = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            grid[i][i] = 1
            grid[0][i] = 1
        for _ in range(m):
            u, v = ac.read_ints()
            grid[u][v] = grid[v][u] = 1
        # 快速幂与最终状态计算
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
        # 模板：线性 DP 使用矩阵幂加速计算
        mod = 10**9 + 7
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
        # 矩阵幂加速
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
        # 模板：脑筋急转弯快速幂计数
        mod = 998244353
        n, k = ac.read_ints()
        ans = pow((pow(2, k, mod) - 1) % mod, n, mod)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8624(ac=FastIO()):
        # 模板：矩阵 DP 与快速幂
        mod = 10**9 + 7
        n, m = ac.read_ints()
        rem = [[0] * 6 for _ in range(6)]
        for _ in range(m):
            i, j = ac.read_ints_minus_one()
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
        # 模板：浮点数快速幂
        if base == 0:
            return 0
        if exponent == 0:
            return 1
        return FastPower().float_fast_pow(base, exponent)
    
    
class TestGeneral(unittest.TestCase):

    def test_fast_power(self):
        fp = FastPower()
        a, b, mod = random.randint(
            1, 123), random.randint(
            1, 1234), random.randint(
            1, 12345)
        assert fp.fast_power_api(a, b, mod) == fp.fast_power(a, b, mod)

        x, n = random.uniform(0, 1), random.randint(1, 1234)
        assert abs(fp.float_fast_pow(x, n) - pow(x, n)) < 1e-5

        mfp = MatrixFastPower()
        mat = [[1, 0, 1], [1, 0, 0], [0, 1, 0]]
        mod = 10 ** 9 + 7
        for _ in range(10):
            n = random.randint(1, 100)
            cur = copy.deepcopy(mat)
            for _ in range(1, n):
                cur = mfp.matrix_mul(cur, mat, mod)
            assert cur == mfp.matrix_pow(
                mat, n, mod) == mfp.matrix_pow(mat, n, mod)

        base = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert mfp.matrix_pow(
            mat, 0, mod) == mfp.matrix_pow(
            mat, 0, mod) == base
        return


if __name__ == '__main__':
    unittest.main()
