import copy
import random
import unittest

from mathmatics.fast_power.template import FastPower, MatrixFastPower


class TestGeneral(unittest.TestCase):

    def test_fast_power(self):
        fp = FastPower()
        a, b, mod = random.randint(
            1, 123), random.randint(
            1, 1234), random.randint(
            1, 12345)
        assert fp.fast_power_api(a, b, mod) == fp.fast_power(a, b, mod)

        x, n = random.uniform(0, 1), random.randint(1, 1234)
        assert abs(fp.float_fast_pow(x, n) - pow(x, n)) < 1e-5

        mfp = MatrixFastPower()
        mat = [[1, 0, 1], [1, 0, 0], [0, 1, 0]]
        mod = 10 ** 9 + 7
        for _ in range(10):
            n = random.randint(1, 100)
            cur = copy.deepcopy(mat)
            for _ in range(1, n):
                cur = mfp.matrix_mul(cur, mat, mod)
            assert cur == mfp.matrix_pow(
                mat, n, mod) == mfp.matrix_pow(mat, n, mod)

        ba = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert mfp.matrix_pow(mat, 0, mod) == mfp.matrix_pow(mat, 0, mod) == ba
        return


if __name__ == '__main__':
    unittest.main()
