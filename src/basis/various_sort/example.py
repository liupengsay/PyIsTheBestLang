import random
import unittest

from src.basis.various_sort.template import VariousSort


class TestGeneral(unittest.TestCase):

    def test_various_sort(self):
        vs = VariousSort()
        n = 200
        for _ in range(n):
            nums = [random.randint(0, n) for _ in range(n)]
            assert vs.defined_sort(nums) == vs.quick_sort_two(nums) == vs.range_merge_to_disjoint_sort(nums) == sorted(
                nums)
        return


if __name__ == '__main__':
    unittest.main()
