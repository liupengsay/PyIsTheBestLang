import math


class ModInverse:
    def __init__(self):
        return

    @staticmethod
    def compute_with_api(a, p):
        assert math.gcd(a, p) == 1
        return pow(a, -1, p)

    @staticmethod
    def ex_gcd(a, b):
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

    def mod_reverse(self, a, p):
        # necessary and sufficient conditions for solving inverse elements
        g, x, y = self.ex_gcd(a, p)
        assert g == 1
        return (x + p) % p
