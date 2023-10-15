

class TestGeneral(unittest.TestCase):
    def test_tree_to_prufer(self):
        ptt = PruferAndTree()
        adj = [[1, 2, 3, 4], [0], [0, 5, 6], [0], [0], [2], [2]]
        code = [0, 0, 0, 2, 2]
        assert ptt.tree_to_prufer(adj, root=6) == code

        ptt = PruferAndTree()
        adj = [[1], [0, 2, 3, 6], [1, 4, 5], [1], [2], [2], [1]]
        code = [1, 1, 2, 2, 1]
        assert ptt.tree_to_prufer(adj, root=1) == code
        return

    def test_prufer_to_tree(self):
        ptt = PruferAndTree()
        code = [0, 0, 0, 2, 2]
        adj = [[1, 2, 3, 4], [0], [0, 5, 6], [0], [0], [2], [2]]
        assert ptt.prufer_to_tree(code, root=6) == adj

        ptt = PruferAndTree()
        code = [1, 1, 2, 2, 1]
        adj = [[1], [0, 2, 3, 6], [1, 4, 5], [1], [2], [2], [1]]
        assert ptt.prufer_to_tree(code, root=1) == adj
        return


if __name__ == '__main__':
    unittest.main()