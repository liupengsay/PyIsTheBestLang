import unittest

from mathmatics.high_precision.template import HighPrecision, FloatToFrac


class TestGeneral(unittest.TestCase):

    def test_high_precision(self):
        hp = HighPrecision()
        assert hp.float_pow("98.999", "5") == "9509420210.697891990494999"

        assert hp.fraction_to_decimal(45, 56) == "0.803(571428)"
        assert hp.fraction_to_decimal(2, 1) == "2.0"
        assert hp.decimal_to_fraction("0.803(571428)") == [45, 56]
        assert hp.decimal_to_fraction("2.0") == [2, 1]
        return

    def test_float_to_frac(self):
        ff = FloatToFrac()
        assert ff.frac_add([1, 2], [1, 3]) == [5, 6]
        assert ff.frac_add([1, 2], [1, -3]) == [1, 6]
        assert ff.frac_add([1, -2], [1, 3]) == [-1, 6]

        assert ff.frac_max([1, 2], [1, 3]) == [1, 2]
        assert ff.frac_min([1, 2], [1, 3]) == [1, 3]

        assert ff.frac_max([1, -2], [1, -3]) == [-1, 3]
        assert ff.frac_min([1, -2], [1, -3]) == [-1, 2]

        assert ff.frac_ceil([2, 3]) == 1
        assert ff.frac_ceil([5, 3]) == 2
        assert ff.frac_ceil([-2, 3]) == 0
        assert ff.frac_ceil([-5, 3]) == -1
        return


if __name__ == '__main__':
    unittest.main()
