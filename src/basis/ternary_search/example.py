import unittest

from src.basis.ternary_search.template import TernarySearch, TriPartPackTriPart


class TestGeneral(unittest.TestCase):

    def test_tri_part_search(self):
        tps = TernarySearch()

        def fun1(x): return (x - 1) * (x - 1)

        assert abs(tps.find_floor_point_float(fun1, -5, 100) - 1) < 1e-5

        def fun2(x): return -(x - 1) * (x - 1)

        assert abs(tps.find_ceil_point_float(fun2, -5, 100) - 1) < 1e-5
        return

    def test_tri_part_pack_tri_part(self):
        tpt = TriPartPackTriPart()
        nodes = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

        def target(x, y): return max([(x - p[0]) ** 2 + (y - p[1]) ** 2 for p in nodes])

        x0, y0, _ = tpt.find_floor_point_float(target, -10, 10, -10, 10)
        assert abs(x0) < 1e-5 and abs(y0) < 1e-5
        return


if __name__ == '__main__':
    unittest.main()
