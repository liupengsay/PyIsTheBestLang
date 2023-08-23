import random
import unittest


class TestGeneral(unittest.TestCase):

    @staticmethod
    def test_print_vp_lst():
        print("\n")
        lst = []
        for num in range(83, 360):
            lst.append(f"https://leetcode.cn/contest/weekly-contest-{num}/")

        for num in range(1, 112):
            lst.append(f"https://leetcode.cn/contest/biweekly-contest-{num}/")

        print(f"Total Contest: {len(lst)}")
        random.shuffle(lst)
        for ls in lst:
            print(ls)
        return


if __name__ == '__main__':
    unittest.main()
