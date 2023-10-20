import unittest

from basis.range.template import Range


class TestGeneral(unittest.TestCase):

    def test_range_cover_count(self):
        rcc = Range()
        lst = [[1, 4], [2, 5], [3, 6], [8, 9]]
        assert rcc.merge(lst) == [[1, 6], [8, 9]]
        return


if __name__ == '__main__':
    unittest.main()
