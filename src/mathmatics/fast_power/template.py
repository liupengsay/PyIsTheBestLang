


class FastPower:
    def __init__(self):
        return

    @staticmethod
    def fast_power_api(a, b, mod):
        return pow(a, b, mod)

    @staticmethod
    def fast_power(a, b, mod):
        a = a % mod
        res = 1
        while b > 0:
            if b & 1:
                res = res * a % mod
            a = a * a % mod
            b >>= 1
        return res

    @staticmethod
    def float_fast_pow(x: float, m: int) -> float:

        if m >= 0:
            res = 1
            while m > 0:
                if m & 1:
                    res *= x
                x *= x
                m >>= 1
            return res
        m = -m
        res = 1
        while m > 0:
            if m & 1:
                res *= x
            x *= x
            m >>= 1
        return 1.0 / res


class MatrixFastPowerFlatten:
    def __init__(self):
        return

    @staticmethod
    def matrix_pow_flatten(base, n, p, mod=10 ** 9 + 7):
        assert len(base) == n * n
        res = [0] * n * n
        ans = [0] * n * n
        for i in range(n):
            ans[i * n + i] = 1
        while p:
            if p & 1:
                for i in range(n):
                    for j in range(n):
                        cur = 0
                        for k in range(n):
                            cur += ans[i * n + k] * base[k * n + j]
                            cur %= mod
                        res[i * n + j] = cur
                for i in range(n):
                    for j in range(n):
                        ans[i * n + j] = res[i * n + j]
            for i in range(n):
                for j in range(n):
                    cur = 0
                    for k in range(n):
                        cur += base[i * n + k] * base[k * n + j]
                        cur %= mod
                    res[i * n + j] = cur
            for i in range(n):
                for j in range(n):
                    base[i * n + j] = res[i * n + j]
            p >>= 1
        return ans

class MatrixFastPowerMin:
    def __init__(self):
        return

    @staticmethod
    def _matrix_mul(a, b):
        n = len(a)
        res = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                res[i][j] = min(max(a[i][k], b[k][j]) for k in range(n))
        return res

    def matrix_pow(self, base, p):
        n = len(base)
        ans = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            ans[i][i] = 0
        while p:
            if p & 1:
                ans = self._matrix_mul(ans, base)
            base = self._matrix_mul(base, base)
            p >>= 1
        return ans


class MatrixFastPower:
    def __init__(self):
        return

    @staticmethod
    def _matrix_mul(a, b, mod=10 ** 9 + 7):
        n = len(a)
        res = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                res[i][j] = sum(a[i][k] * b[k][j] for k in range(n)) % mod
        return res

    def matrix_pow(self, base, p, mod=10 ** 9 + 7):
        n = len(base)
        ans = [[0] * n for _ in range(n)]
        for i in range(n):
            ans[i][i] = 1
        while p:
            if p & 1:
                ans = self._matrix_mul(ans, base, mod)
            base = self._matrix_mul(base, base, mod)
            p >>= 1
        return ans
