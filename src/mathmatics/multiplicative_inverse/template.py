import math


class MultiplicativeInverse:
    def __init__(self):
        return

    @staticmethod
    def compute_with_api(a, p):
        assert math.gcd(a, p) == 1
        return pow(a, -1, p)

    def ex_gcd(self, a, b):
        if b == 0:
            return 1, 0, a
        x, y, q = self.ex_gcd(b, a % b)
        x, y = y, (x - (a // b) * y)
        return x, y, q

    # 扩展欧几里得求逆元
    def mod_reverse(self, a, p):
        # necessary and sufficient conditions for solving inverse elements
        assert math.gcd(a, p) == 1
        x, y, q = self.ex_gcd(a, p)
        return (x + p) % p  # pow(a, -1, p)
