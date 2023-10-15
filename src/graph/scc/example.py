

class TestGeneral(unittest.TestCase):

    def test_directed_graph(self):
        # 有向无环图
        edge = [[1, 2], [], [3], []]
        n = 4
        kosaraju = Kosaraju(n, edge)
        assert len(set(kosaraju.color)) == 4
        tarjan = Tarjan(edge)
        assert len(tarjan.scc) == 4

        # 有向有环图
        edge = [[1, 2], [2], [0, 3], []]
        n = 4
        kosaraju = Kosaraju(n, edge)
        assert len(set(kosaraju.color)) == 2
        tarjan = Tarjan(edge)
        assert len(tarjan.scc) == 2
        return

    def test_undirected_graph(self):
        # 无向有环图
        edge = [[1, 2], [0, 2, 3], [0, 1], [1, 4], [3]]
        n = 5
        kosaraju = Kosaraju(n, edge)
        assert len(set(kosaraju.color)) == 1
        tarjan = Tarjan(edge)
        assert len(tarjan.scc) == 1
        return


if __name__ == '__main__':
    unittest.main()
