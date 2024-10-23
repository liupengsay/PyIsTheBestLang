import unittest

from src.string.lyndon_decomposition.template import LyndonDecomposition


class TestGeneral(unittest.TestCase):
    def test_solve_by_duval(self):
        ld = LyndonDecomposition()
        assert ld.solve_by_duval("ababa") == ["ab", "ab", "a"]
        return

    def test_min_cyclic_string(self):
        ld = LyndonDecomposition()
        assert ld.min_cyclic_string("ababa") == "aabab"
        assert ld.min_express("ababa")[1] == "aabab"
        return


if __name__ == '__main__':
    unittest.main()
