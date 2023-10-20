class MultiplicativeInverse:
    def __init__(self):
        return

    @staticmethod
    def compute_with_api(a, p):
        return pow(a, -1, p)

    # 扩展欧几里得求逆元
    def ex_gcd(self, a, b):
        if b == 0:
            return 1, 0, a
        x, y, q = self.ex_gcd(b, a % b)
        x, y = y, (x - (a // b) * y)
        return x, y, q

    # 扩展欧几里得求逆元
    def mod_reverse(self, a, p):
        # 要求a与p互质
        x, y, q = self.ex_gcd(a, p)
        if q != 1:
            raise Exception("No solution.")
        return (x + p) % p
