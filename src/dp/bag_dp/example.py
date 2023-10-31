import unittest

from src.dp.bag_dp.template import BagDP


class TestGeneral(unittest.TestCase):

    def test_bag_dp(self):
        bd = BagDP()
        for num in range(1, 100000):
            lst1 = bd.bin_split_1(num)
            lst2 = bd.bin_split_2(num)
            assert sum(lst1) == num == sum(lst2)
            assert len(lst1) == len(lst2)
        return


if __name__ == '__main__':
    unittest.main()
