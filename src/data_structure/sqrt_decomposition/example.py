import unittest

from src.data_structure.sqrt_decomposition.template import BlockSize


class TestGeneral(unittest.TestCase):

    def test_block_size(self):
        bs = BlockSize()
        for x in range(1, 10 ** 4 + 1):
            bs.get_divisor_split(x)
        cnt, seg = bs.get_divisor_split(100)
        assert cnt == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 5, 8, 17, 50]
        assert seg == [[1, 4], [5, 5], [6, 5], [6, 6], [7, 6], [7, 7], [8, 7], [8, 8], [9, 9], [10, 10], [11, 11],
                       [12, 12], [13, 14], [15, 16], [17, 20], [21, 25], [26, 33], [34, 50], [51, 100]]
        return


if __name__ == '__main__':
    unittest.main()
