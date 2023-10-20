import unittest

from dp.linear_dp.template import LinearDP


class TestGeneral(unittest.TestCase):

    def test_linear_dp(self):
        ld = LinearDP()
        nums = [6, 3, 5, 2, 1, 6, 8, 9]
        assert ld.liner_dp_template(nums) == 4
        return


if __name__ == '__main__':
    unittest.main()
