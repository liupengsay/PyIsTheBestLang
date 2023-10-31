import unittest

from src.mathmatics.geometry.template import Geometry


class TestGeneral(unittest.TestCase):

    def test_geometry(self):
        gm = Geometry()
        assert gm.compute_square_point(1, 1, 6, 6) == ((6.0, 1.0), (1.0, 6.0))
        assert gm.compute_square_point(0, 0, 0, 2) == ((1.0, 1.0), (-1.0, 1.0))

        assert gm.compute_triangle_area(0, 0, 2, 0, 1, 1) == 1.0
        return


if __name__ == '__main__':
    unittest.main()
