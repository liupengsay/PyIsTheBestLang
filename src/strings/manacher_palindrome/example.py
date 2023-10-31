import unittest

from src.strings.manacher_palindrome.template import ManacherPlindrome


class TestGeneral(unittest.TestCase):

    def test_manacher_palindrome(self):
        s = "abccba"
        mp = ManacherPlindrome()
        start, end = mp.palindrome(s)
        assert start == [[0, 5], [1, 4], [2, 3], [3], [4], [5]]
        assert end == [[0], [1], [2], [2, 3], [1, 4], [0, 5]]
        return


if __name__ == '__main__':
    unittest.main()
