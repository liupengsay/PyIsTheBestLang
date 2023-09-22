
import math
import sys
import unittest
from decimal import Decimal, getcontext, MAX_PREC
from typing import List

from src.fast_io import FastIO

getcontext().prec = MAX_PREC
sys.set_int_max_str_digits(0)  # 力扣大数的范围坑

"""
算法：大数分解、素数判断、高精度计算、使用分数代替浮点数运算
功能：xxx
题目：

===================================力扣===================================
166. 分数到小数（https://leetcode.cn/problems/fraction-to-recurring-decimal/）经典分数转换为有理数无限循环小数
172. 阶乘后的零（https://leetcode.cn/problems/factorial-trailing-zeroes/）阶乘后缀0的个数
1883. 准时抵达会议现场的最小跳过休息次数（https://leetcode.cn/problems/minimum-skips-to-arrive-at-meeting-on-time/description/）经典二维矩阵DP使用分数进行高精度浮点数计算
2117. 一个区间内所有数乘积的缩写（https://leetcode.cn/problems/abbreviating-the-product-of-a-range/）大数计算或者前后缀模拟计算
972. 相等的有理数（https://leetcode.cn/problems/equal-rational-numbers/）有理数转为分数判断

===================================洛谷===================================
P2388 阶乘之乘（https://www.luogu.com.cn/problem/P2388）阶乘之乘后缀0的个数

P1920 成功密码（https://www.luogu.com.cn/problem/P1920）预估高精度计算与公式 -ln(1-x) = sum(x**i/i for in range(1, n+1)) 其中 n 趋近于无穷
P1729 计算e（https://www.luogu.com.cn/problem/P1729）高精度计算e小数位
P1727 计算π（https://www.luogu.com.cn/problem/P1727）高精度计算π小数位
P1517 高精求小数幂（https://www.luogu.com.cn/record/list?user=739032&status=12&page=5）高精度计算小数的幂值
P2394 yyy loves Chemistry I（https://www.luogu.com.cn/problem/P2394）高精度计算
P2393 yyy loves Maths II（https://www.luogu.com.cn/problem/P2393）高精度计算

P2399 non hates math（https://www.luogu.com.cn/problem/P2399）小数有理数转换为最简分数
P1530 [USACO2.4]分数化小数 Fractions to Decimals（https://www.luogu.com.cn/problem/P1530）分数化为小数

===================================AtCoder===================================
E - Double Factorial（https://atcoder.jp/contests/abc148/tasks/abc148_e）奇数阶乘与偶数阶乘的尾随零个数

参考：OI WiKi（xx）
"""


class HighPrecision:
    def __init__(self):
        return

    @staticmethod
    def factorial_to_factorial(n):
        # 模板: 计算1!*2!***n!的后缀0个数
        ans = 0
        num = 5
        while num <= n:
            ans += num * (n // num) * (n // num - 1) // 2
            ans += (n // num) * (n % num + 1)
            num *= 5
        return ans

    @staticmethod
    def factorial_to_zero(n):
        # 模板: 计算n!的后缀0个数
        ans = 0
        while n:
            ans += n // 5
            n //= 5
        return ans

    @staticmethod
    def float_pow(r, n):
        # 高精度计算小数的幂值
        ans = (Decimal(r) ** int(n)).normalize()
        ans = "{:f}".format(ans)
        return ans

    @staticmethod
    def fraction_to_decimal(numerator: int, denominator: int) -> str:
        # 分数转换为有理数或者无限循环小数
        if numerator % denominator == 0:
            return str(numerator // denominator) + ".0"
        ans = []
        if numerator * denominator < 0:
            ans.append("-")
        numerator = abs(numerator)
        denominator = abs(denominator)

        ans.append(str(numerator // denominator))
        numerator %= denominator
        ans.append(".")
        reminder = numerator % denominator
        dct = dict()
        while reminder and reminder not in dct:
            dct[reminder] = len(ans)
            reminder *= 10
            ans.append(str(reminder // denominator))
            reminder %= denominator
        if reminder in dct:
            ans.insert(dct[reminder], "(")
            ans.append(")")
        return "".join(ans)

    @staticmethod
    def decimal_to_fraction(st):
        # 模板：小数转分数
        def sum_fraction(tmp):
            # 分数相加
            mu = tmp[0][1]
            for ls in tmp[1:]:
                mu = math.lcm(mu, ls[1])
            zi = sum(ls[0] * mu // ls[1] for ls in tmp)
            mz = math.gcd(mu, zi)
            return [zi // mz, mu // mz]

        # 有理数或无限循环小数转换为分数
        if "." in st:
            lst = st.split(".")
            integer = [int(lst[0]), 1] if lst[0] else [0, 1]
            if "(" not in lst[1]:
                non_repeat = [int(lst[1]), 10 ** len(lst[1])
                              ] if lst[1] else [0, 1]
                repeat = [0, 1]
            else:
                pre, post = lst[1].split("(")
                non_repeat = [int(pre), 10 ** len(pre)] if pre else [0, 1]
                post = post[:-1]
                repeat = [int(post), int("9" * len(post)) * 10 ** len(pre)]
        else:

            integer = [int(st), 1]
            non_repeat = [0, 1]
            repeat = [0, 1]
        return sum_fraction([integer, non_repeat, repeat])


class FloatToFrac:
    def __init__(self):
        # 将浮点数运算转换为分数计算
        return

    @staticmethod
    def frac_add(frac1: List[int], frac2: List[int]) -> List[int]:
        # 要求分母a1与a2均不为0
        b1, a1 = frac1
        b2, a2 = frac2
        a = math.lcm(a1, a2)
        b = b1 * (a // a1) + b2 * (a // a2)
        g = math.gcd(b, a)
        b //= g
        a //= g
        if a < 0:
            a *= -1
            b *= -1
        return [b, a]

    @staticmethod
    def frac_max(frac1: List[int], frac2: List[int]) -> List[int]:
        # 要求分母a1与a2均不为0
        b1, a1 = frac1
        b2, a2 = frac2
        if a1 < 0:
            a1 *= -1
            b1 *= -1
        if a2 < 0:
            a2 *= -1
            b2 *= -1
        if b1 * a2 < b2 * a1:
            return [b2, a2]
        return [b1, a1]

    @staticmethod
    def frac_min(frac1: List[int], frac2: List[int]) -> List[int]:
        # 要求分母a1与a2均不为0
        b1, a1 = frac1
        b2, a2 = frac2
        if a1 < 0:
            a1 *= -1
            b1 *= -1
        if a2 < 0:
            a2 *= -1
            b2 *= -1
        if b1 * a2 > b2 * a1:
            return [b2, a2]
        return [b1, a1]

    @staticmethod
    def frac_ceil(frac: List[int]) -> int:
        # 要求分母a1与a2均不为0
        b, a = frac
        return math.ceil(b / a)


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_148e(ac=FastIO()):
        # 模板：奇数阶乘与偶数阶乘的尾随零个数
        n = ac.read_int()
        if n % 2:
            ac.st(0)
        else:
            ans = HighPrecision().factorial_to_zero(n//10) + n//10
            ac.st(ans)
        return

    @staticmethod
    def lc_172(n):
        # 模板: n!的后缀零个数
        return HighPrecision().factorial_to_zero(n)

    @staticmethod
    def lc_972(s: str, t: str) -> bool:
        # 模板：有理数转为分数判断
        hp = HighPrecision()
        return hp.decimal_to_fraction(s) == hp.decimal_to_fraction(t)

    @staticmethod
    def lg_p2238(ac=FastIO()):
        # 模板: 1!*2!*...*n!的后缀零个数
        n = ac.read_int()
        ac.st(HighPrecision().factorial_to_factorial(n))
        return

    @staticmethod
    def lg_p2399(ac=FastIO()):
        # 模板：有理数转最简分数
        s = ac.read_str()
        a, b = HighPrecision().decimal_to_fraction(s)
        ac.st(f"{a}/{b}")
        return

    @staticmethod
    def lc_2217(left: int, right: int) -> str:
        # 模板：大数计算或者前后缀模拟计算
        mod = 10 ** 20
        base = 10 ** 10
        zero = 0
        suffix = 1
        for x in range(left, right + 1):
            suffix *= x
            while suffix % 10 == 0:
                zero += 1
                suffix //= 10

            suffix %= mod

        prefix = 1
        for x in range(left, right + 1):
            prefix *= x
            while prefix % 10 == 0:
                prefix //= 10

            while prefix > mod:
                prefix //= 10

        if prefix >= base:
            return str(prefix)[:5] + "..." + str(suffix)[-5:] + "e" + str(zero)
        else:
            return str(prefix) + "e" + str(zero)

    @staticmethod
    def lg_p1530(ac=FastIO()):
        # 模板：最简分数转化为有理数
        n, d = ac.read_ints()
        ans = HighPrecision().fraction_to_decimal(n, d)
        while ans:
            ac.st(ans[:76])
            ans = ans[76:]
        return

    @staticmethod
    def lc_1883_1(dist: List[int], speed: int, hours: int) -> int:

        # 模板：经典二维矩阵DP使用分数进行高精度浮点数计算
        n = len(dist)
        if sum(dist) > hours * speed:
            return -1

        ff = FloatToFrac()
        dp = [[[hours * 2, 1] for _ in range(n + 1)] for _ in range(n)]
        dp[0][0] = [0, 1]
        for i in range(n - 1):
            dp[i + 1][0] = ff.frac_add(dp[i][0], [ff.frac_ceil([dist[i], speed]), 1])
            for j in range(1, i + 2):
                pre1 = [ff.frac_ceil(ff.frac_add(dp[i][j], [dist[i], speed])), 1]
                pre2 = ff.frac_add(dp[i][j - 1], [dist[i], speed])
                dp[i + 1][j] = ff.frac_min(pre1, pre2)
        for j in range(n + 1):
            cur = ff.frac_add(dp[n - 1][j], [dist[n - 1], speed])
            if cur[0] <= hours * cur[1]:
                return j
        return -1
    
    @staticmethod
    def lc_1883_2(dist: List[int], speed: int, hours: int) -> int:

        # 模板：经典二维矩阵DP使用分数进行高精度浮点数计算
        cost = [Decimal(d)/Decimal(speed) for d in dist]
        n = len(dist)
        dp = [[hours*2]*(n+1) for _ in range(n+1)]
        dp[0][0] = 0
        for i in range(1, n):
            # 使用浮点数计算
            dp[i][0] = dp[i-1][0] + math.ceil(cost[i-1])
            for j in range(1, i):
                a, b = dp[i-1][j-1] + cost[i-1], math.ceil(dp[i-1][j] + cost[i-1])
                dp[i][j] = a if a < b else b

            dp[i][i] = dp[i-1][i-1] + cost[i-1]

        for j in range(n+1):
            if dp[n-1][j] + cost[-1] <= hours:
                return j
        return -1
    
    
class TestGeneral(unittest.TestCase):

    def test_high_precision(self):
        hp = HighPrecision()
        assert hp.float_pow("98.999", "5") == "9509420210.697891990494999"

        assert hp.fraction_to_decimal(45, 56) == "0.803(571428)"
        assert hp.fraction_to_decimal(2, 1) == "2.0"
        assert hp.decimal_to_fraction("0.803(571428)") == [45, 56]
        assert hp.decimal_to_fraction("2.0") == [2, 1]
        return

    def test_float_to_frac(self):
        ff = FloatToFrac()
        assert ff.frac_add([1, 2], [1, 3]) == [5, 6]
        assert ff.frac_add([1, 2], [1, -3]) == [1, 6]
        assert ff.frac_add([1, -2], [1, 3]) == [-1, 6]

        assert ff.frac_max([1, 2], [1, 3]) == [1, 2]
        assert ff.frac_min([1, 2], [1, 3]) == [1, 3]

        assert ff.frac_max([1, -2], [1, -3]) == [-1, 3]
        assert ff.frac_min([1, -2], [1, -3]) == [-1, 2]

        assert ff.frac_ceil([2, 3]) == 1
        assert ff.frac_ceil([5, 3]) == 2
        assert ff.frac_ceil([-2, 3]) == 0
        assert ff.frac_ceil([-5, 3]) == -1
        return


if __name__ == '__main__':
    unittest.main()
