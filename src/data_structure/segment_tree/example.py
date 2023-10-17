import random
import unittest

from data_structure.segment_tree.template import SegmentTreePointAddSumMaxMin, RangeAscendRangeMax, \
    SegmentTreeRangeUpdateMax, SegmentTreeRangeUpdateMin, RangeDescendRangeMin, \
    SegmentTreeRangeUpdateQuerySumMinMax, SegmentTreeRangeChangeQuerySumMinMax, SegmentTreeRangeUpdateSubConSum, \
    SegmentTreeRangeUpdateQuery, SegmentTreeRangeUpdateSum, SegmentTreeRangeAddSum, RangeOrRangeAnd


class TestGeneral(unittest.TestCase):

    def test_segment_tree_range_add_sum(self):
        low = 0
        high = 10 ** 9 + 7
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        segment_tree = SegmentTreeRangeAddSum()
        for i in range(n):
            segment_tree.update(i, i, low, high, nums[i], 1)

        for _ in range(n):
            # 区间增加值
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(low, high)
            segment_tree.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            assert segment_tree.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])

            # 单点增加值
            left = random.randint(0, n - 1)
            right = left
            num = random.randint(low, high)
            segment_tree.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert segment_tree.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])
        return

    def test_segment_tree_range_update_sum(self):
        low = 0
        high = 10 ** 9 + 7
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        segment_tree = SegmentTreeRangeUpdateSum()
        for i in range(n):
            segment_tree.update(i, i, low, high, nums[i], 1)

        for _ in range(n):
            # 区间增加值
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(low, high)
            segment_tree.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            assert segment_tree.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])

            # 单点增加值
            left = random.randint(0, n - 1)
            right = left
            num = random.randint(low, high)
            segment_tree.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert segment_tree.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])
        return

    def test_range_or_range_and(self):
        low = 0
        high = (1 << 31) - 1
        n = 1000
        nums = [random.randint(low, high) for _ in range(n)]
        segment_tree = RangeOrRangeAnd(n)
        segment_tree.build(nums)
        for _ in range(1000):
            ll = random.randint(0, n-1)
            rr = random.randint(ll, n-1)
            num = random.randint(low, high)
            for i in range(ll, rr+1):
                nums[i] |= num
            segment_tree.range_or(ll, rr, num)

            ll = random.randint(0, n-1)
            rr = random.randint(ll, n-1)
            res = high
            for i in range(ll, rr+1):
                res &= nums[i]
            assert res == segment_tree.range_and(ll, rr)
        assert nums == segment_tree.get()
        return

    def test_segment_tree_point_add_sum_max_min(self):
        low = 0
        high = 10000

        nums = [random.randint(low, high) for _ in range(high)]
        staasmm = SegmentTreePointAddSumMaxMin(high)
        for i in range(high):
            staasmm.add(1, low, high, i + 1, nums[i])

        for _ in range(high):
            # 单点进行增减值
            i = random.randint(0, high - 1)
            num = random.randint(low, high)
            nums[i] += num
            staasmm.add(1, low, high, i + 1, num)

            # 查询区间和、最大值、最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert staasmm.query_sum(
                1, low, high, left + 1, right + 1) == sum(nums[left:right + 1])
            assert staasmm.query_max(
                1, low, high, left + 1, right + 1) == max(nums[left:right + 1])
            assert staasmm.query_min(
                1, low, high, left + 1, right + 1) == min(nums[left:right + 1])

            # 查询单点和、最大值、最小值
            left = random.randint(0, high - 1)
            right = left
            assert staasmm.query_sum(
                1, low, high, left + 1, right + 1) == sum(nums[left:right + 1])
            assert staasmm.query_max(
                1, low, high, left + 1, right + 1) == max(nums[left:right + 1])
            assert staasmm.query_min(
                1, low, high, left + 1, right + 1) == min(nums[left:right + 1])

        return

    def test_segment_tree_range_add_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        segment_tree = RangeAscendRangeMax(high)
        segment_tree.build(nums)
        assert segment_tree.get() == nums
        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            segment_tree.range_ascend(left, right, num)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.range_max(left, right) == max(nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            segment_tree.range_ascend(left, right, num)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num
            assert segment_tree.range_max(left, right) == max(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.range_max(left, right) == max(nums[left:right + 1])
        assert segment_tree.get() == nums
        return

    def test_segment_tree_range_update_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low + 1, high) for _ in range(high)]
        segment_tree = SegmentTreeRangeUpdateMax()
        for i in range(high):
            segment_tree.update(i, i, low, high, nums[i], 1)
            assert segment_tree.query(i, i, low, high, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low + 1, high)
            for i in range(left, right + 1):
                nums[i] = num
            segment_tree.update(left, right, low, high, num, 1)
            assert segment_tree.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low + 1, high)
            segment_tree.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert segment_tree.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query(left, right, low, high, 1) == max(
                nums[left:right + 1])
        return

    def test_segment_tree_range_update_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low + 1, high) for _ in range(high)]
        segment_tree = SegmentTreeRangeUpdateMin()
        for i in range(high):
            segment_tree.update(i, i, low, high, nums[i], 1)
            assert segment_tree.query(i, i, low, high, 1) == min(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low + 1, high)
            for i in range(left, right + 1):
                nums[i] = num
            segment_tree.update(left, right, low, high, num, 1)
            assert segment_tree.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low + 1, high)
            segment_tree.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert segment_tree.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query(left, right, low, high, 1) == min(
                nums[left:right + 1])
        return

    def test_range_descend_range_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        segment_tree = RangeDescendRangeMin(high)
        segment_tree.build(nums)

        for _ in range(high):
            # 区间更新与查询最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            segment_tree.range_descend(left, right, num)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.range_min(left, right) == min(nums[left:right + 1])

            # 单点更新与查询最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            segment_tree.range_descend(left, right, num)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            assert segment_tree.range_min(left, right) == min(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.range_min(left, right) == min(nums[left:right + 1])

        assert segment_tree.get() == nums
        return

    def test_segment_tree_range_change_query_sum_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        n = len(nums)
        segment_tree = SegmentTreeRangeUpdateQuerySumMinMax(n)
        segment_tree.build(nums)
        for i in range(high):
            assert segment_tree.query_min(i, i, low, high - 1, 1) == min(nums[i:i + 1])
            assert segment_tree.query_sum(i, i, low, high - 1, 1) == sum(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            segment_tree.update_range(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            segment_tree.update_point(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

        assert segment_tree.get_all_nums() == nums
        return

    def test_segment_tree_range_update_query_sum_min_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        segment_tree = SegmentTreeRangeUpdateQuerySumMinMax(len(nums))
        segment_tree.build(nums)

        assert segment_tree.query_min(0, high - 1, low, high - 1, 1) == min(nums)
        assert segment_tree.query_max(0, high - 1, low, high - 1, 1) == max(nums)
        assert segment_tree.query_sum(0, high - 1, low, high - 1, 1) == sum(nums)

        for i in range(high):
            assert segment_tree.query_min(i, i, low, high - 1, 1) == min(nums[i:i + 1])
            assert segment_tree.query_sum(i, i, low, high - 1, 1) == sum(nums[i:i + 1])
            assert segment_tree.query_max(i, i, low, high - 1, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            segment_tree.update_range(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值使用 update
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            segment_tree.update_range(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值使用 update_point
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            segment_tree.update_point(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])
        assert segment_tree.get_all_nums() == nums
        return

    def test_segment_tree_range_change_query_sum_min_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        segment_tree = SegmentTreeRangeChangeQuerySumMinMax(nums)

        assert segment_tree.query_min(0, high - 1, low, high - 1, 1) == min(nums)
        assert segment_tree.query_max(0, high - 1, low, high - 1, 1) == max(nums)
        assert segment_tree.query_sum(0, high - 1, low, high - 1, 1) == sum(nums)

        for i in range(high):
            assert segment_tree.query_min(i, i, low, high - 1, 1) == min(nums[i:i + 1])
            assert segment_tree.query_sum(i, i, low, high - 1, 1) == sum(nums[i:i + 1])
            assert segment_tree.query_max(i, i, low, high - 1, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            segment_tree.change(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            segment_tree.change(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_max(left, right, low, high - 1, 1) == max(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_max(left, right, low, high - 1, 1) == max(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值 change_point
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            segment_tree.change_point(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_max(left, right, low, high - 1, 1) == max(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert segment_tree.query_max(left, right, low, high - 1, 1) == max(
                nums[left:right + 1])
            assert segment_tree.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

        segment_tree.get_all_nums()
        assert segment_tree.nums == nums
        return

    def test_segment_tree_range_sub_con_max(self):

        def check(lst):
            pre = ans = lst[0]
            for x in lst[1:]:
                pre = pre + x if pre + x > x else x
                ans = ans if ans > pre else pre
            return ans

        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        segment_tree = SegmentTreeRangeUpdateSubConSum(nums)

        assert segment_tree.query_max(0, high - 1, low, high - 1, 1)[0] == check(nums)

        for _ in range(high):
            # 区间更新值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            segment_tree.update(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert segment_tree.query_max(left, right, low, high - 1, 1)[0] == check(nums[left:right + 1])
        return

    def test_segment_tree_range_change_point_query(self):
        low = 0
        high = 10 ** 9 + 7
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        segment_tree = SegmentTreeRangeUpdateQuery(n)
        for i in range(n):
            segment_tree.update(i, i, 0, n - 1, nums[i], 1)

        for _ in range(10):
            # 区间修改值
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(low, high)
            segment_tree.update(left, right, 0, n - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            for i in range(n):
                assert segment_tree.query(i, i, 0, n - 1, 1) == nums[i]
        return


if __name__ == '__main__':
    unittest.main()
