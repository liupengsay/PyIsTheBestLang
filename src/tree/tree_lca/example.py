import unittest

from src.tree.tree_dp.template import WeightedTree


class TestGeneral(unittest.TestCase):

    def test_tree_ancestor(self):
        parent = [-1, 0, 0, 1, 2]
        n = len(parent)
        graph = WeightedTree(n)
        for i in range(n):
            if parent[i] != -1:
                graph.add_undirected_edge(parent[i], i, 1)
        graph.lca_build_with_multiplication()
        assert graph.lca_get_kth_ancestor(4, 3) == -1
        assert graph.lca_get_kth_ancestor(4, 2) == 0
        assert graph.lca_get_kth_ancestor(4, 1) == 2
        assert graph.lca_get_kth_ancestor(4, 0) == 4
        assert graph.lca_get_lca_between_nodes(3, 4) == 0
        assert graph.lca_get_lca_between_nodes(2, 4) == 2
        assert graph.lca_get_lca_between_nodes(3, 1) == 1
        assert graph.lca_get_lca_between_nodes(3, 2) == 0
        assert graph.lca_get_lca_and_dist_between_nodes(0, 0)[1] == 0
        assert graph.lca_get_lca_and_dist_between_nodes(0, 4)[1] == 2
        assert graph.lca_get_lca_and_dist_between_nodes(3, 4)[1] == 4
        assert graph.lca_get_lca_and_dist_between_nodes(1, 0)[1] == 1
        assert graph.lca_get_lca_and_dist_between_nodes(2, 3)[1] == 3
        return


if __name__ == '__main__':
    unittest.main()
