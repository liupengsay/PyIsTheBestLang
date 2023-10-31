import unittest

from src.mathmatics.extend_crt.template import ExtendCRT, CRT


class TestGeneral(unittest.TestCase):

    def test_crt(self):
        pairs = [(3, 1), (5, 1), (7, 2)]
        crt = CRT()
        assert crt.chinese_remainder(pairs) == 16

        exc = ExtendCRT()
        pairs = [(6, 11), (9, 25), (17, 33)]
        assert exc.pipline(pairs)[0] == 809
        return


if __name__ == '__main__':
    unittest.main()
