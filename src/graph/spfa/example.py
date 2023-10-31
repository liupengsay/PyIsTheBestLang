import unittest

from src.graph.spfa.template import SPFA


class TestGeneral(unittest.TestCase):

    def test_spfa(self):
        dct = [{1: 5, 2: 1}, {3: 4}, {3: 2}, {}]
        spfa = SPFA()
        res, dis, cnt = spfa.negative_circle(dct)
        assert res == "NO"
        assert dis == [0, 5, 1, 3]
        assert cnt == [0, 1, 1, 2]

        dct = [{1: 5, 2: 1}, {3: 4}, {3: 2}, {2: -4}]
        spfa = SPFA()
        res, _, _ = spfa.negative_circle(dct)
        assert res == "YES"
        return

    def test_spfa_cnt(self):
        dct = [{1: 3, 2: 2}, {3: 4}, {3: 1}, {}]
        spfa = SPFA()
        assert spfa.count_shortest_path(dct) == [1, 3, 2, 14]
        return


if __name__ == '__main__':
    unittest.main()
