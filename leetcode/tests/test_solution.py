import logging
import unittest
from leetcode.solution import Solution


class Test(unittest.TestCase):
    def test_num_str(self):
        s = "3[b]2[ac]"
        self.assertEqual("bbbacac", Solution.num_str(s))

        s = "3[b2[ac]]"
        self.assertEqual("bacacbacacbacac", Solution.num_str(s))

        s = "3[b]2[ac]ef"
        self.assertEqual("bbbacacef", Solution.num_str(s))

        s = "ef3[b]2[ac]"
        self.assertEqual("efbbbacac", Solution.num_str(s))
        return


if __name__ == '__main__':
    unittest.main()
