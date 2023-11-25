import unittest

from src.graph.euler_path.template import DirectedEulerPath


class TestGeneral(unittest.TestCase):

    def test_euler_path(self):
        pairs = [[1, 2], [2, 3], [3, 4], [4, 3], [3, 2], [2, 1]]
        pairs = [[x - 1, y - 1] for x, y in pairs]
        ep = DirectedEulerPath(4, pairs)
        ans = [[x + 1, y + 1] for x, y in ep.paths]
        assert ans == [[1, 2], [2, 3], [3, 4], [4, 3], [3, 2], [2, 1]]

        pairs = [[1, 3], [2, 1], [4, 2], [3, 3], [1, 2], [3, 4]]
        pairs = [[x - 1, y - 1] for x, y in pairs]
        ep = DirectedEulerPath(4, pairs)
        ans = [x + 1 for x in ep.nodes]
        assert ans == [1, 2, 1, 3, 3, 4, 2]
        return


if __name__ == '__main__':
    unittest.main()
