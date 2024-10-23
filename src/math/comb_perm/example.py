import math
import unittest

from src.math.comb_perm.template import Combinatorics


class TestGeneral(unittest.TestCase):
    def test_comb_perm(self):
        for x in range(10):
            n = 10 ** 3 + x * 10 ** 2
            mod = 10 ** 9 + 7
            cb = Combinatorics(n, mod)
            for i in range(1, n + 1):
                assert pow(i, -1, mod) == cb.inv[i] == cb.inverse(i)
                assert math.factorial(i) % mod == cb.perm[i] == cb.factorial(i)
                assert math.comb(n, i) % mod == cb.comb(n, i)
        return


if __name__ == '__main__':
    unittest.main()
