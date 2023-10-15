

class TestGeneral(unittest.TestCase):
    def test_undirected_graph(self):
        # 无向无环图
        edge = [[1, 2], [0, 3], [0, 3], [1, 2]]
        n = 4
        ta = TarjanUndirected()
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert not cut_edge
        assert not cut_node
        assert sub_group == [[0, 1, 2, 3]]

        # 无向有环图
        edge = [[1, 2, 3], [0, 2], [0, 1], [0]]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[0, 3]]
        assert cut_node == [0]
        assert sub_group == [[0, 1, 2], [3]]

        # 无向有环图
        edge = [[1, 2], [0, 2], [0, 1, 3], [2]]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[2, 3]]
        assert cut_node == [2]
        assert sub_group == [[0, 1, 2], [3]]

        # 无向有自环图
        edge = [[1, 2], [0, 2], [0, 1, 3], [2, 3]]
        n = 4
        cut_edge, cut_node, sub_group = ta.check_graph(edge, n)
        assert cut_edge == [[2, 3]]
        assert cut_node == [2]
        assert sub_group == [[0, 1, 2], [3]]
        return


if __name__ == '__main__':
    unittest.main()
