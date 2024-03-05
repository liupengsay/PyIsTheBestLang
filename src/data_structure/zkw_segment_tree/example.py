import random
import unittest
from math import inf
from operator import add

from src.data_structure.segment_tree.template import PointSetRangeSum, RangeAddPointGet, RangeAddRangeSumMinMax
from src.data_structure.tree_array.template import PointAddRangeSum as PointAddRangeSumTA
from src.data_structure.zkw_segment_tree.template import (
    PointSetAddRangeSum as PointSetRangeSumZKW,
    RangeAddPointGet as RangeAddPointGetZKW,
    LazySegmentTree as LazySegmentTreeZKW)


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

            i = random.randint(0, n - 1)
            num = random.randint(low, high)
            nums[i] += num
            tree.point_add(i, num)

            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            ans = tree.range_sum(ll, rr)
            assert ans == sum(nums[ll:rr + 1])

        assert nums == tree.get()
        return

    def test_range_add_point_get(self):
        random.seed(2024)
        low = 1
        high = 100
        n = 1000000
        nums = [random.randint(low, high) for _ in range(n)]
        tree = RangeAddPointGet(n)
        tree.build(nums)
        assert nums == tree.get()
        for _ in range(10000):
            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            num = random.randint(low, high)
            for i in range(ll, rr + 1):
                nums[i] += num
            tree.range_add(ll, rr, num)

            ll = random.randint(0, n - 1)
            ans = tree.point_get(ll)
            assert ans == nums[ll]

        assert nums == tree.get()
        return

    def test_range_add_point_get_zkw(self):
        random.seed(2024)
        low = 1
        high = 100
        n = 1000000
        nums = [random.randint(low, high) for _ in range(n)]
        tree = RangeAddPointGetZKW(n)
        tree.build(nums)
        assert nums == tree.get()
        for _ in range(10000):
            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            num = random.randint(low, high)
            for i in range(ll, rr + 1):
                nums[i] += num
            tree.range_add(ll, rr, num)

            ll = random.randint(0, n - 1)
            ans = tree.point_get(ll)
            assert ans == nums[ll]

        assert nums == tree.get()
        return

    def test_range_add_range_sum_min_max(self):
        low = -10000
        high = 50000
        nums = [random.randint(low, high) for _ in range(high)]
        tree = RangeAddRangeSumMinMax(high)
        tree.build(nums)

        assert tree.range_min(0, high - 1) == min(nums)
        assert tree.range_max(0, high - 1) == max(nums)
        assert tree.range_sum(0, high - 1) == sum(nums)

        for _ in range(high):

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            tree.range_add(left, right, num)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            tree.range_add(left, right, num)
            for i in range(left, right + 1):
                nums[i] += num
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

        assert tree.get() == nums
        return

    def test_range_add_range_sum_min_max_zkw(self):
        low = -10000
        high = 50000

        def add_only(a, b, c):
            return a + b

        def add_with_length(a, b, c):
            return a + b * c

        nums = [random.randint(low, high) for _ in range(high)]
        tree_max = LazySegmentTreeZKW(high, max, -inf, add_only, add, 0)
        tree_min = LazySegmentTreeZKW(high, min, inf, add_only, add, 0)
        tree_sum = LazySegmentTreeZKW(high, add, 0, add_with_length, add, 0)
        tree_max.build(nums)
        tree_min.build(nums)
        tree_sum.build(nums)

        assert tree_min.range_query(0, high - 1) == min(nums)
        assert tree_max.range_query(0, high - 1) == max(nums)
        assert tree_sum.range_query(0, high - 1) == sum(nums)

        for _ in range(high):

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            tree_max.range_update(left, right, num)
            tree_min.range_update(left, right, num)
            tree_sum.range_update(left, right, num)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree_min.range_query(left, right) == min(
                nums[left:right + 1])
            assert tree_max.range_query(left, right) == max(
                nums[left:right + 1])
            assert tree_sum.range_query(left, right) == sum(
                nums[left:right + 1])

        assert tree_min.get() == nums
        assert tree_max.get() == nums
        assert tree_sum.get() == nums
        return

    def test_range_set_range_sum_min_max_zkw(self):
        low = -10000
        high = 50000

        def add_only(a, b, c):
            return b

        def add_with_length(a, b, c):
            return b * c

        def merge_tag(tag1, tag2):
            return tag1

        nums = [random.randint(low, high) for _ in range(high)]
        tree_max = LazySegmentTreeZKW(high, max, -inf, add_only, merge_tag, inf)
        tree_min = LazySegmentTreeZKW(high, min, inf, add_only, merge_tag, inf)
        tree_sum = LazySegmentTreeZKW(high, add, 0, add_with_length, merge_tag, inf)
        tree_max.build(nums)
        tree_min.build(nums)
        tree_sum.build(nums)

        assert tree_min.range_query(0, high - 1) == min(nums)
        assert tree_max.range_query(0, high - 1) == max(nums)
        assert tree_sum.range_query(0, high - 1) == sum(nums)

        for _ in range(high):

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            tree_max.range_update(left, right, num)
            tree_min.range_update(left, right, num)
            tree_sum.range_update(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree_min.range_query(left, right) == min(
                nums[left:right + 1])
            assert tree_max.range_query(left, right) == max(
                nums[left:right + 1])
            assert tree_sum.range_query(left, right) == sum(
                nums[left:right + 1])

        assert tree_min.get() == nums
        assert tree_max.get() == nums
        assert tree_sum.get() == nums
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
