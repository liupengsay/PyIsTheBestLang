import math


class Combinatorics:
    def __init__(self, n, mod):
        n += 10
        self.perm = [1] * n
        self.rev = [1] * n
        self.mod = mod
        for i in range(1, n):
            # (i!) % mod
            self.perm[i] = self.perm[i - 1] * i
            self.perm[i] %= self.mod
        self.rev[-1] = self.mod_reverse(self.perm[-1], self.mod)  # equal to pow(self.perm[-1], -1, self.mod)
        for i in range(n - 2, 0, -1):
            self.rev[i] = (self.rev[i + 1] * (i + 1) % mod)  # pow(i!, -1, mod)
        self.fault = [0] * n
        self.fault_perm()
        return

    def ex_gcd(self, a, b):
        if b == 0:
            return 1, 0, a
        else:
            x, y, q = self.ex_gcd(b, a % b)
            x, y = y, (x - (a // b) * y)
            return x, y, q

    def mod_reverse(self, a, p):
        assert math.gcd(a, p) == 1
        x, y, q = self.ex_gcd(a, p)
        return (x + p) % p

    def comb(self, a, b):
        if a < b:
            return 0
        # C(a, b) % mod
        res = self.perm[a] * self.rev[b] * self.rev[a - b]
        return res % self.mod

    def factorial(self, a):
        # (a!) % mod
        res = self.perm[a]
        return res % self.mod

    def fault_perm(self):
        # number of fault combinations
        self.fault[0] = 1
        self.fault[2] = 1
        for i in range(3, len(self.fault)):
            self.fault[i] = (i - 1) * (self.fault[i - 1] + self.fault[i - 2])
            self.fault[i] %= self.mod
        return

    def inv(self, n):
        # pow(n, -1, mod)
        return self.perm[n - 1] * self.rev[n] % self.mod

    def catalan(self, n):
        return (self.comb(2 * n, n) - self.comb(2 * n, n - 1)) % self.mod


class Lucas:
    def __init__(self):
        # Comb(a,b) % p
        return

    def lucas(self, n, m, p):
        # math.comb(n, m) % p where p is prime
        if m == 0:
            return 1
        return ((math.comb(n % p, m % p) % p) * self.lucas(n // p, m // p, p)) % p

    @staticmethod
    def comb(n, m, p):
        # comb(n, m ) % p
        ans = 1
        for x in range(n - m + 1, n + 1):
            ans *= x
            ans %= p
        for x in range(1, m + 1):
            ans *= pow(x, -1, p)
            ans %= p
        return ans

    def lucas_iter(self, n, m, p):
        # math.comb(n, m) % p where p is prime
        if m == 0:
            return 1
        stack = [[n, m]]
        dct = dict()
        while stack:
            n, m = stack.pop()
            if n >= 0:
                if m == 0:
                    dct[(n, m)] = 1
                    continue
                stack.append([~n, m])
                stack.append([n // p, m // p])
            else:
                n = ~n
                dct[(n, m)] = (self.comb(n % p, m % p, p) % p) * dct[(n // p, m // p)] % p
        return dct[(n, m)]

    @staticmethod
    def extend_lucas(self, n, m, p):
        # math.comb(n, m) % p where p is not necessary prime
        return
