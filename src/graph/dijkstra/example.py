import unittest

from src.graph.dijkstra.template import Dijkstra


class TestGeneral(unittest.TestCase):

    def test_dijkstra(self):
        djk = Dijkstra()
        dct = [[[1, 1], [2, 4]], [[2, 2]], []]
        assert djk.get_shortest_path(dct, 0) == [0, 1, 3]
        return


if __name__ == '__main__':
    unittest.main()
