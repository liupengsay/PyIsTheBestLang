from functools import reduce


class CRT:
    def __init__(self):
        return

    @staticmethod
    def chinese_remainder(pairs):
        mod_list, remainder_list = [p[0] for p in pairs], [p[1] for p in pairs]
        mod_product = reduce(lambda x, y: x * y, mod_list)
        mi_list = [mod_product // x for x in mod_list]
        mi_inverse = [ExtendCRT().extend_gcd(mi_list[i], mod_list[i])[0] for i in range(len(mi_list))]
        x = 0
        for i in range(len(remainder_list)):
            x += mi_list[i] * mi_inverse[i] * remainder_list[i]
            x %= mod_product
        return x


class ExtendCRT:
    def __init__(self):
        return

    def gcd(self, a, b):
        if b == 0:
            return a
        return self.gcd(b, a % b)

    def lcm(self, a, b):
        return a * b // self.gcd(a, b)

    def extend_gcd(self, a, b):
        if b == 0:
            return 1, 0
        x, y = self.extend_gcd(b, a % b)
        return y, x - a // b * y

    def uni(self, p, q):
        r1, m1 = p
        r2, m2 = q

        d = self.gcd(m1, m2)
        assert (r2 - r1) % d == 0  # else without solution
        l1, l2 = self.extend_gcd(m1 // d, m2 // d)

        return (r1 + (r2 - r1) // d * l1 * m1) % self.lcm(m1, m2), self.lcm(m1, m2)

    def pipline(self, eq):
        return reduce(self.uni, eq)
