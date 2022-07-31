import logging
import unittest
from leetcode.solution import Solution


class Test(unittest.TestCase):
    def test_bin(self):
        n = 2**8
        i = 0
        dct = {}
        for i in range(1, n):
            res = []
            r = i
            while i < n:
                res.append(bin(i))
                i += i&(-i)
            dct[r] = res

        i = 0
        dct2 = {}
        for i in range(1, 15):
            res = []
            r = i
            while i > 0:
                res.append(bin(i))
                i -= i&(-i)
            dct2[r] = res

        res = [ i + (i & -i) for i in range(1, 14)]

        bins = [bin(i) for i in range(1, 15)]
        bins_neg = [bin(-i) for i in range(1, 15)]
        bins_and = [bin(i + i & -i) for i in range(1, 10)]
        for i in range(1, n):
            # print(bin(i), bin(-i), bin(i &(-i)))
            while i < n:
                print(i, i+ i & (-i))
            print('\n')
        return

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
