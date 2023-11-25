import unittest

from src.data_structure.block_size.template import BlockSize


class TestGeneral(unittest.TestCase):

    def test_block_size(self):
        bs = BlockSize()
        for x in range(1, 10 ** 4 + 1):
            bs.get_divisor_split(x)
        pass


if __name__ == '__main__':
    unittest.main()
