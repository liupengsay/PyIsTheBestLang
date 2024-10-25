class GcdLike:

    def __init__(self):
        return

    @staticmethod
    def extend_gcd(a, b):
        sub = dict()
        stack = [(a, b, 0)]
        while stack:
            a, b, s = stack.pop()
            if a == 0:
                sub[(a, b)] = (b, 0, 1) if b >= 0 else (-b, 0, -1)
                continue
            if s == 0:
                stack.append((a, b, 1))
                stack.append((b % a, a, 0))
            else:
                gcd, x, y = sub[(b % a, a)]
                sub[(a, b)] = (gcd, y - (b // a) * x, x) if gcd >= 0 else (-gcd, -y + (b // a) * x, -x)
                assert gcd == a * (y - (b // a) * x) + b * x
        return sub[(a, b)]

    @staticmethod
    def binary_gcd(a, b):
        if a == 0:
            return abs(b)
        if b == 0:
            return abs(a)
        a, b = abs(a), abs(b)
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
    def general_gcd(x, y):
        while y:
            x, y = y, x % y
        return abs(x)

    def mod_reverse(self, a, p):
        g, x, y = self.extend_gcd(a, p)
        assert g == 1  # necessary of pow(a, -1, p)
        return (x + p) % p

    def solve_equation(self, a, b, n=1):
        """
        a*x+b*y=n
        (a*x)%b=n
        """
        gcd, x, y = self.extend_gcd(a, b)
        assert a * x + b * y == gcd
        if n % gcd:
            return []
        x0 = x * (n // gcd)
        y0 = y * (n // gcd)
        # xt = x0 + b // gcd * t (t=0,1,2,3,...)
        # yt = y0 - a // gcd * t (t=0,1,2,3,...)
        return [gcd, x0, y0]

    @staticmethod
    def add_to_n(n):
        # minimum times to make a == n or b == n by change [a, b] to [a + b, b] or [a, a + b] from [1, 1]
        if n == 1:
            return 0

        def gcd_minus(a, b, c):
            nonlocal ans
            if c >= ans or not b:
                return
            if b == 1:
                ans = ans if ans < c + a - 1 else c + a - 1
                return
            # reverse_thinking
            gcd_minus(b, a % b, c + a // b)
            return

        ans = n - 1
        for i in range(1, n):
            gcd_minus(n, i, 0)
        return ans
