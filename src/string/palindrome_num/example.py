import unittest

from src.string.palindrome_num.template import PalindromeNum


class TestGeneral(unittest.TestCase):

    def test_palindrome_num(self):
        pn = PalindromeNum()
        assert pn.get_palindrome_num_1(12) == pn.get_palindrome_num_2(12)

        n = "44"
        nums = pn.get_recent_palindrome_num(n)
        nums = [num for num in nums if num > int(n)]
        assert min(nums) == 55
        return


if __name__ == '__main__':
    unittest.main()
