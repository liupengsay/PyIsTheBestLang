import unittest

from src.greedy.brain_storming.template import BrainStorming


class TestGeneral(unittest.TestCase):

    def test_brain_storming(self):
        bs = BrainStorming()
        n, m = 4, 20
        nums = [1, 2, 5, 10]
        assert bs.minimal_coin_need(n, m, nums) == 5
        return


if __name__ == '__main__':
    unittest.main()
