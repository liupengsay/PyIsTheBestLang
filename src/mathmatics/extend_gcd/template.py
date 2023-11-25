import math

from src.utils.fast_io import FastIO
import math

from src.utils.fast_io import FastIO


class ExtendGcd:
    def __init__(self):
        return

    def extend_gcd(self, a, b):
        # 扩展欧几里得算法，求解ax+by=1，返回gcd(a,b)与x与y（也可以求ax % b=1的解）
        if a == 0:
            return b, 0, 1
        else:
            gcd, x, y = self.extend_gcd(b % a, a)
            return gcd, y - (b // a) * x, x

    def solve_equal(self, a, b, m=1):
        # 模板：扩展gcd求解ax+by=m方程组的所有解
        gcd, x0, y0 = self.extend_gcd(a, b)
        # 方程有解当且仅当m是gcd(a,b)的倍数
        assert a * x0 + b * y0 == 1

        # 方程组的解初始值则为
        x1 = x0 * (m // gcd)
        y1 = y0 * (m // gcd)

        # 所有解为下面这些，可进一步求解正数负数解
        # x = x1+b//gcd*t(t=0,1,2,3,...)
        # y = y1-a//gcd*t(t=0,1,2,3,...)
        return [gcd, x1, y1]

    @staticmethod
    def binary_gcd(a, b):
        # 模板：二进制gcd，使用二进制求两个正数的gcd
        assert a > 0 and b > 0
        c = 1
        while a - b:
            if a & 1:
                if b & 1:
                    if a > b:
                        a = (a - b) >> 1
                    else:
                        b = (b - a) >> 1
                else:
                    b = b >> 1
            else:
                if b & 1:
                    a = a >> 1
                else:
                    c = c << 1
                    b = b >> 1
                    a = a >> 1
        return c * a

    @staticmethod
    def ac_4296(ac=FastIO()):
        # 模板：扩展欧几里得求解ax+by=n的非负整数解
        n, a, b = [ac.read_int() for _ in range(3)]
        g = math.gcd(a, b)
        if n % g:
            ac.st("NO")
        else:
            # 求解ax+by=n且x>=0和y>=0
            gcd, x1, y1 = ExtendGcd().solve_equal(a, b, n)
            low = math.ceil((-x1 * gcd) / b)
            high = (y1 * gcd) // a
            # low<=t<=high
            if low <= high:
                x = x1 + (b // gcd) * low
                ac.st("YES")
                ac.lst([x, (n - a * x) // b])
            else:
                ac.st("NO")
        return
