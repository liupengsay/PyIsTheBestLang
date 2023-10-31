import unittest

from src.graph.lgv.template import LGV


class TestGeneral(unittest.TestCase):
    def test_euler_phi(self):
        lgv = LGV()
        assert lgv.get_result(1, 1, 2, 2) == 2
        return


if __name__ == '__main__':
    unittest.main()
