import unittest


class TestGeneral(unittest.TestCase):

    def test_solution(self):
        assert Solution().getMaximumConsecutive([1, 3]) == 2

        return


if __name__ == '__main__':
    unittest.main()
