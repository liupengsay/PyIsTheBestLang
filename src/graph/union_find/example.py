import unittest

from graph.union_find.template import UnionFind, PersistentUnionFind


class TestGeneral(unittest.TestCase):

    def test_union_find(self):
        uf = UnionFind(5)
        for i, j in [[0, 1], [1, 2]]:
            uf.union(i, j)
        assert uf.part == 3
        return

    def test_persistent_union_find(self):
        # 在线根据历史版本时间戳查询
        n = 3
        puf = PersistentUnionFind(n)
        edge_list = [[0, 1, 2], [1, 2, 4], [2, 0, 8], [1, 0, 16]]
        edge_list.sort(key=lambda item: item[2])
        for x, y, tm in edge_list:
            puf.union(x, y, tm)
        queries = [[0, 1, 2], [0, 2, 5]]
        assert [puf.is_connected(x, y, tm) for x, y, tm in queries] == [False, True]


if __name__ == '__main__':
    unittest.main()
