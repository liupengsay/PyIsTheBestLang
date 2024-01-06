class ExtendGcd:
    def __init__(self):
        return

    def extend_gcd(self, a, b):
        sub = dict()
        stack = [(a, b)]
        while stack:
            a, b = stack.pop()
            if a == 0:
                sub[(a, b)] = (b, 0, 1)
                continue
            if b >= 0:
                stack.append((a, ~b))
                stack.append((b % a, a))
            else:
                b = ~b
                gcd, x, y = sub[(b % a, a)]
                sub[(a, b)] = (gcd, y - (b // a) * x, x)
                assert gcd == a * (y - (b // a) * x) + b * x
        return sub[(a, b)]

    def solve_equal(self, a, b, m=1):
        # 模板：扩展gcd求解ax+by=m方程组的所有解
        gcd, x0, y0 = self.extend_gcd(a, b)
        # 方程有解当且仅当m是gcd(a,b)的倍数
        if m % gcd:
            return []
        # 方程组的解初始值则为
        x1 = x0 * (m // gcd)
        y1 = y0 * (m // gcd)

        # 所有解为下面这些，可进一步求解正数负数解
        # x = x1+b//gcd*t(t=0,1,2,3,...)
        # y = y1-a//gcd*t(t=0,1,2,3,...)
        return [gcd, x1, y1]

    @staticmethod
    def binary_gcd(a, b):
        # bin_gcd|，二进制求两个正数的gcd
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
