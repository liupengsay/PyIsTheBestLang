import unittest

from src.mathmatics.peishu_theorem.template import PeiShuTheorem


class TestGeneral(unittest.TestCase):

    def test_peishu_theorem(self):
        lst = [4059, -1782]
        pst = PeiShuTheorem().get_lst_gcd(lst)
        assert pst == 99
        return


if __name__ == '__main__':
    unittest.main()
