import random
import unittest

from src.data_structure.segment_tree.template import PointSetRangeSum
from src.data_structure.tree_array.template import PointAddRangeSum as PointAddRangeSumTA
from src.data_structure.zkw_segment_tree.template import PointSetRangeSum as PointSetRangeSumZKW
from src.data_structure.zkw_segment_tree.template import PointSetRangeSumStack as PointSetRangeSumSTACK


class TestGeneral(unittest.TestCase):

    def test_point_set_range_sum(self):
        random.seed(2024)
        low = 1
        high = 100
        n = 100000
        nums = [random.randint(low, high) for _ in range(n)]
        tree = PointSetRangeSum(n, 0)
        tree.build(nums)
        for _ in range(10000):
            i = random.randint(0, n - 1)
            num = random.randint(low, high)
            nums[i] = num
            tree.point_set(i, num)

            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            ans = tree.range_sum(ll, rr)
            assert ans == sum(nums[ll:rr + 1])

        assert nums == tree.get()
        return

    def test_point_set_range_sum_zkw(self):
        random.seed(2024)
        low = 1
        high = 100
        n = 100000
        nums = [random.randint(low, high) for _ in range(n)]
        tree = PointSetRangeSumZKW(n, 0)
        tree.build(nums)
        assert nums == tree.get()
        for _ in range(10000):
            i = random.randint(0, n - 1)
            num = random.randint(low, high)
            nums[i] = num
            tree.point_set(i, num)

            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            ans = tree.range_sum(ll, rr)
            assert ans == sum(nums[ll:rr + 1])

        assert nums == tree.get()
        return

    def test_point_set_range_sum_stack(self):
        random.seed(2024)
        low = 1
        high = 100
        n = 100000
        nums = [random.randint(low, high) for _ in range(n)]
        tree = PointSetRangeSumSTACK(n, 0)
        tree.build(nums)
        assert nums == tree.get()
        for _ in range(10000):
            i = random.randint(0, n - 1)
            num = random.randint(low, high)
            nums[i] = num
            tree.point_set(i, num)
            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            ans = tree.range_sum(ll, rr)
            assert ans == sum(nums[ll:rr + 1])

        assert nums == tree.get()
        return

    def test_point_set_range_sum_ta(self):
        random.seed(2024)
        low = 1
        high = 100
        n = 100000
        nums = [random.randint(low, high) for _ in range(n)]
        tree = PointAddRangeSumTA(n, 0)
        tree.build(nums)
        for _ in range(10000):
            i = random.randint(0, n - 1)
            num = random.randint(low, high)
            nums[i] += num
            tree.point_add(i + 1, num)

            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            ans = tree.range_sum(ll + 1, rr + 1)
            assert ans == sum(nums[ll:rr + 1])

        assert nums == tree.get()
        return


if __name__ == '__main__':
    unittest.main()
