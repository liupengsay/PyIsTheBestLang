import unittest

from src.search.dfs.template import DFS, DfsEulerOrder


class TestGeneral(unittest.TestCase):

    def test_dfs(self):
        dfs = DFS()
        dct = [[1, 2], [0, 3], [0, 4], [1], [2]]
        start, end = dfs.gen_bfs_order_iteration([d[::-1] for d in dct])
        assert start == [x - 1 for x in [1, 2, 4, 3, 5]]
        assert end == [b - 1 for _, b in [[1, 5], [2, 3], [4, 5], [3, 3], [5, 5]]]
        return

    def test_dfs_euler(self):
        dct = [[1, 2], [3, 4], [0, 5], [1], [1, 6], [2], [4]]
        dfs = DfsEulerOrder(dct)
        assert dfs.order_to_node == [0, 1, 3, 4, 6, 2, 5]
        assert dfs.euler_order == [0, 1, 3, 1, 4, 6, 4, 1, 0, 2, 5, 2, 0]
        return


if __name__ == '__main__':
    unittest.main()
