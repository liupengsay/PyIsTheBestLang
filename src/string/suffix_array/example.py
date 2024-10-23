import random
import unittest

from src.string.suffix_array.template import SuffixArray


class TestGeneral(unittest.TestCase):

    def test_suffix_array(self):
        sa = SuffixArray()
        for x in range(7):
            lst = [random.randint(0, 25) for _ in range(10 ** x)]
            sa.build(lst[:], 26)
        return


if __name__ == '__main__':
    unittest.main()
