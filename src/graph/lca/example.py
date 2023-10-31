import unittest

from src.graph.lca.template import TreeAncestor


class TestGeneral(unittest.TestCase):

    def test_tree_ancestor(self):
        parent = [-1, 0, 0, 1, 2]
        n = len(parent)
        edges = [[] for _ in range(n)]
        for i in range(n):
            if parent[i] != -1:
                edges[i].append(parent[i])
                edges[parent[i]].append(i)
        tree = TreeAncestor(edges)
        assert tree.get_kth_ancestor(4, 3) == -1
        assert tree.get_kth_ancestor(4, 2) == 0
        assert tree.get_kth_ancestor(4, 1) == 2
        assert tree.get_kth_ancestor(4, 0) == 4
        assert tree.get_lca(3, 4) == 0
        assert tree.get_lca(2, 4) == 2
        assert tree.get_lca(3, 1) == 1
        assert tree.get_lca(3, 2) == 0
        assert tree.get_dist(0, 0) == 0
        assert tree.get_dist(0, 4) == 2
        assert tree.get_dist(3, 4) == 4
        assert tree.get_dist(1, 0) == 1
        assert tree.get_dist(2, 3) == 3
        return


if __name__ == '__main__':
    unittest.main()
