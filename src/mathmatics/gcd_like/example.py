import math
import random
import unittest

from src.mathmatics.gcd_like.template import GcdLike
from src.utils.fast_io import SEED

random.seed(SEED)


class TestGeneral(unittest.TestCase):

    def test_gcd_like(self):
        gl = GcdLike()
        n = 10 ** 5
        for _ in range(1000):
            a = random.randint(-n, n)
            b = random.randint(-n, n)
            gcd1, x, y = gl.extend_gcd(a, b)
            gcd2 = gl.binary_gcd(a, b)
            gcd3 = gl.general_gcd(a, b)
            gcd4 = math.gcd(a, b)
            assert gcd1 == gcd2 == gcd3 == gcd4
            assert a * x + b * y == gcd1

        for a in range(-1000, 1000):
            for b in range(-1000, 1000):
                gcd1, x, y = gl.extend_gcd(a, b)
                gcd2 = gl.binary_gcd(a, b)
                gcd3 = gl.general_gcd(a, b)
                gcd4 = math.gcd(a, b)
                assert gcd1 == gcd2 == gcd3 == gcd4
                assert a * x + b * y == gcd1
        return

    def test_gcd_like_extend_gcd(self):
        gl = GcdLike()
        n = 10 ** 5
        for _ in range(1000):
            a = random.randint(-n, n)
            b = random.randint(-n, n)
            gl.extend_gcd(a, b)

        for a in range(-1000, 1000):
            for b in range(-1000, 1000):
                gl.extend_gcd(a, b)
        return

    def test_gcd_like_binary_gcd(self):
        gl = GcdLike()
        n = 10 ** 5
        for _ in range(1000):
            a = random.randint(-n, n)
            b = random.randint(-n, n)
            gl.binary_gcd(a, b)

        for a in range(-1000, 1000):
            for b in range(-1000, 1000):
                gl.binary_gcd(a, b)
        return

    def test_gcd_like_general_gcd(self):
        gl = GcdLike()
        n = 10 ** 5
        for _ in range(1000):
            a = random.randint(-n, n)
            b = random.randint(-n, n)
            gl.general_gcd(a, b)

        for a in range(-1000, 1000):
            for b in range(-1000, 1000):
                gl.general_gcd(a, b)
        return

    def test_gcd_like_math_gcd(self):
        n = 10 ** 5
        for _ in range(1000):
            a = random.randint(-n, n)
            b = random.randint(-n, n)
            math.gcd(a, b)

        for a in range(-1000, 1000):
            for b in range(-1000, 1000):
                math.gcd(a, b)
        return

    def test_gcd_like_mod_reverse(self):
        n = 10 ** 5
        gl = GcdLike()
        for _ in range(1000):
            a = random.randint(-n, n)
            b = random.randint(-n, n)
            if math.gcd(a, b) == 1 and b:
                assert gl.mod_reverse(a, b) == pow(a, -1, b)

        for a in range(-1000, 1000):
            for b in range(-1000, 1000):
                math.gcd(a, b)
                if math.gcd(a, b) == 1 and b:
                    assert gl.mod_reverse(a, b) == pow(a, -1, b)
        return


if __name__ == '__main__':
    unittest.main()
