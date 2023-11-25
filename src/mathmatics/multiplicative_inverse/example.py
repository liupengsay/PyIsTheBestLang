import math
import random
import unittest

from src.mathmatics.multiplicative_inverse.template import MultiplicativeInverse


class TestGeneral(unittest.TestCase):

    def test_multiplicative_inverse(self):
        mt = MultiplicativeInverse()
        assert mt.mod_reverse(10, 13) == 4
        assert mt.compute_with_api(10, 13) == 4
        assert mt.mod_reverse(10, 1) == 0
        assert mt.compute_with_api(10, 1) == 0
        mod = 10 ** 9 + 7
        for _ in range(1000):
            num = random.randint(1, 10 ** 9)
            assert pow(num, -1, mod) == mt.mod_reverse(num, mod)

            a, b = random.randint(10, 1000), random.randint(10, 10000)
            if math.gcd(a, b) == 1:
                assert pow(a, -1, b) == mt.mod_reverse(a, b)
        return


if __name__ == '__main__':
    unittest.main()
