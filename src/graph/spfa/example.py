import unittest

from src.graph.spfa.template import SPFA


class TestGeneral(unittest.TestCase):

    def test_spfa(self):
        return

    def test_spfa_cnt(self):
        dct = [{1: 3, 2: 2}, {3: 4}, {3: 1}, {}]
        spfa = SPFA()
        assert spfa.count_shortest_path(dct) == [1, 3, 2, 14]
        return


if __name__ == '__main__':
    unittest.main()
