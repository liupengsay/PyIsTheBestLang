import unittest

from graph.minimum_spanning_tree.template import MinimumSpanningTree


class TestGeneral(unittest.TestCase):

    def test_minimum_spanning_tree(self):
        n = 3
        edges = [[0, 1, 2], [1, 2, 3], [2, 0, 4]]
        mst = MinimumSpanningTree(edges, n)
        assert mst.cost == 5

        n = 4
        edges = [[0, 1, 2], [1, 2, 3], [2, 0, 4]]
        mst = MinimumSpanningTree(edges, n)
        assert mst.cost == -1
        return


if __name__ == '__main__':
    unittest.main()
