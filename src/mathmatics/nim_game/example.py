import unittest

from src.mathmatics.nim_game.template import Nim


class TestGeneral(unittest.TestCase):
    def test_nim_game(self):
        nim = Nim([0, 2, 3])
        assert nim.gen_result()
        return


if __name__ == '__main__':
    unittest.main()
