class Combinatorics:
    def __init__(self, n, mod):
        assert mod > n
        self.n = n
        self.mod = mod

        self.perm = [1]
        self.rev = [1]
        self.inv = [0]
        self.fault = [0]

        self.build_perm()
        self.build_rev()
        self.build_inv()
        self.build_fault()
        return

    def build_perm(self):
        self.perm = [1] * (self.n + 1)  # (i!) % mod
        for i in range(1, self.n + 1):
            self.perm[i] = self.perm[i - 1] * i % self.mod
        return

    def build_rev(self):
        self.rev = [1] * (self.n + 1)  # pow(i!, -1, mod)
        self.rev[-1] = pow(self.perm[-1], -1, self.mod)  # GcdLike().mod_reverse(self.perm[-1], self.mod)
        for i in range(self.n - 1, 0, -1):
            self.rev[i] = (self.rev[i + 1] * (i + 1) % self.mod)  # pow(i!, -1, mod)
        return

    def build_inv(self):
        self.inv = [0] * (self.n + 1)  # pow(i, -1, mod)
        self.inv[1] = 1
        for i in range(2, self.n + 1):
            self.inv[i] = (self.mod - self.mod // i) * self.inv[self.mod % i] % self.mod
        return

    def build_fault(self):
        self.fault = [0] * (self.n + 1)  # fault permutation
        self.fault[0] = 1
        self.fault[2] = 1
        for i in range(3, self.n + 1):
            self.fault[i] = (i - 1) * (self.fault[i - 1] + self.fault[i - 2])
            self.fault[i] %= self.mod
        return

    def comb(self, a, b):
        if a < b:
            return 0
        res = self.perm[a] * self.rev[b] * self.rev[a - b]  # comb(a, b) % mod = (a!/(b!(a-b)!)) % mod
        return res % self.mod

    def factorial(self, a):
        res = self.perm[a]  # (a!) % mod
        return res % self.mod

    def inverse(self, n):
        res = self.perm[n - 1] * self.rev[n] % self.mod  # pow(n, -1, mod)
        return res

    def catalan(self, n):
        res = (self.comb(2 * n, n) - self.comb(2 * n, n - 1)) % self.mod
        return res


class Lucas:
    def __init__(self):
        # comb(a,b) % p
        return

    @staticmethod
    def comb(n, m, p):
        # comb(n, m) % p
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
                stack.append((~n, m))
                stack.append((n // p, m // p))
            else:
                n = ~n
                dct[(n, m)] = (self.comb(n % p, m % p, p) % p) * dct[(n // p, m // p)] % p
        return dct[(n, m)]

    @staticmethod
    def extend_lucas(self, n, m, p):
        # math.comb(n, m) % p where p is not necessary prime
        return
