import random
import unittest
from math import inf
from operator import add

from src.data_structure.segment_tree.template import PointSetRangeSum, RangeAddPointGet, RangeAddRangeSumMinMax
from src.data_structure.tree_array.template import PointAddRangeSum as PointAddRangeSumTA
from src.data_structure.zkw_segment_tree.template import (
    PointSetAddRangeSum as PointSetRangeSumZKW,
    RangeAddPointGet as RangeAddPointGetZKW,
    LazySegmentTree as LazySegmentTreeZKW, LazySegmentTreeLength)


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

    def test_range_set_reverse_range_sum_longest_con_sub_sum(self):
        random.seed(2024)
        high = 5 * 100000

        # start, end, cover_0, cover_1, length, sum
        # start>0 1 start<0 0
        # 0-change 1-change 2-reverse 3-null
        nums = [random.randint(0, 1) for _ in range(high)]

        def merge_cover_tag(cover, tag):
            length = cover[-2]
            if tag == 0:
                return -length, -length, length, 0, length, 0
            elif tag == 1:
                return length, length, 0, length, length, length
            elif tag == 2:  # 2
                start, end, cover_zero, cover_one, length, tot = cover
                return -start, -end, cover_one, cover_zero, length, length - tot
            return cover

        def combine_cover_cover(cover1, cover2):
            start1, end1, cover_zero1, cover_one1, length1, sum1 = cover1
            if cover1 == cover_initial:
                return cover2
            if cover2 == cover_initial:
                return cover1
            start2, end2, cover_zero2, cover_one2, length2, sum2 = cover2
            start = start1
            if (start1 == length1 or start1 == -length1) and start1 * start2 > 0:
                start += start2
            end = end2
            if (start2 == length2 or start2 == -length2) and end1 * end2 > 0:
                end += end1

            cover_zero = cover_zero1 if cover_zero1 > cover_zero2 else cover_zero2
            if start < 0 and -start > cover_zero:
                cover_zero = -start
            if end < 0 and -end > cover_zero:
                cover_zero = -end

            cover_one = cover_one1 if cover_one1 > cover_one2 else cover_one2
            if start > 0 and start > cover_one:
                cover_one = start
            if end > 0 and end > cover_one:
                cover_one = end

            if end1 > 0 and start2 > 0 and end1 + start2 > cover_one:
                cover_one = end1 + start2
            if end1 < 0 and start2 < 0 and -end1 - start2 > cover_zero:
                cover_zero = -end1 - start2

            length = length1 + length2
            return start, end, cover_zero, cover_one, length, sum1 + sum2

        def merge_tag_tag(tag1, tag2):
            if tag1 == 3:
                return tag2
            if tag1 <= 1:
                return tag1
            # tag1 = 2
            if tag2 <= 1:
                tag2 = 1 - tag2
            elif tag2 == 2:
                tag2 = 3
            else:
                tag2 = 2
            return tag2

        cover_initial = (0, 0, 0, 0, 0, 0)
        tag_initial = 3

        tree = LazySegmentTreeLength(high, combine_cover_cover, cover_initial, merge_cover_tag, merge_tag_tag,
                                     tag_initial)
        tree.build(nums)

        def check(tmp):
            ans = pre = 0
            for x in tmp:
                if x:
                    pre += 1
                else:
                    ans = ans if ans > pre else pre
                    pre = 0
            ans = ans if ans > pre else pre
            return ans

        assert [ls[-1] for ls in tree.get()] == nums
        assert tree.range_query(0, high - 1)[3] == check(nums)
        assert tree.cover[1][3] == check(nums)
        for _ in range(100):
            op = random.randint(0, 3)
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            if op == 0:
                tree.range_update(left, right, op)
                for i in range(left, right + 1):
                    nums[i] = 0
            elif op == 1:
                tree.range_update(left, right, op)
                for i in range(left, right + 1):
                    nums[i] = 1
            elif op == 2:
                tree.range_update(left, right, op)
                for i in range(left, right + 1):
                    nums[i] = 1 - nums[i]
            elif op == 3:
                assert tree.range_query(left, right)[-1] == sum(nums[left:right + 1])
            else:
                assert tree.range_query(left, right)[3] == check(nums[left:right + 1])
            assert [ls[-1] for ls in tree.get()] == nums

        assert [int(ls[0] == 1) for ls in tree.get()] == nums
        return


if __name__ == '__main__':
    unittest.main()
