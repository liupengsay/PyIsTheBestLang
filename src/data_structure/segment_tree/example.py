import random
import unittest

from data_structure.segment_tree.template import SegmentTreePointAddSumMaxMin, SegmentTreeRangeAddMax, \
    SegmentTreeRangeUpdateMax, SegmentTreeRangeUpdateMin, SegmentTreeUpdateQueryMin, \
    SegmentTreeRangeUpdateQuerySumMinMax, SegmentTreeRangeChangeQuerySumMinMax, SegmentTreeRangeUpdateSubConSum, \
    SegmentTreeRangeUpdateQuery, SegmentTreeRangeUpdateSum, SegmentTreeRangeAddSum


class TestGeneral(unittest.TestCase):

    def test_segment_tree_range_add_sum(self):
        low = 0
        high = 10 ** 9 + 7
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        stra = SegmentTreeRangeAddSum()
        for i in range(n):
            stra.update(i, i, low, high, nums[i], 1)

        for _ in range(n):
            # 区间增加值
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            assert stra.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])

            # 单点增加值
            left = random.randint(0, n - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert stra.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])
        return

    def test_segment_tree_range_update_sum(self):
        low = 0
        high = 10 ** 9 + 7
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        stra = SegmentTreeRangeUpdateSum()
        for i in range(n):
            stra.update(i, i, low, high, nums[i], 1)

        for _ in range(n):
            # 区间增加值
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            assert stra.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])

            # 单点增加值
            left = random.randint(0, n - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query(left, right, low, high, 1) == sum(
                nums[left:right + 1])
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
        stra = SegmentTreeRangeAddMax(high)
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])
        return

    def test_segment_tree_range_update_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low + 1, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateMax()
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low + 1, high)
            for i in range(left, right + 1):
                nums[i] = num
            stra.update(left, right, low, high, num, 1)
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low + 1, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == max(
                nums[left:right + 1])
        return

    def test_segment_tree_range_update_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low + 1, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateMin()
        for i in range(high):
            stra.update(i, i, low, high, nums[i], 1)
            assert stra.query(i, i, low, high, 1) == min(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low + 1, high)
            for i in range(left, right + 1):
                nums[i] = num
            stra.update(left, right, low, high, num, 1)
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low + 1, high)
            stra.update(left, right, low, high, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query(left, right, low, high, 1) == min(
                nums[left:right + 1])
        return

    def test_segment_tree_update_query_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeUpdateQueryMin(high)
        stra.build(nums)

        for _ in range(high):
            # 区间更新与查询最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            stra.update_range(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_range(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])

            # 单点更新与查询最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            stra.update_point(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            assert stra.query_point(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_range(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])

        assert stra.get() == nums[:]
        return

    def test_segment_tree_range_change_query_sum_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        n = len(nums)
        stra = SegmentTreeRangeUpdateQuerySumMinMax(n)
        stra.build(nums)
        for i in range(high):
            assert stra.query_min(i, i, low, high - 1, 1) == min(nums[i:i + 1])
            assert stra.query_sum(i, i, low, high - 1, 1) == sum(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            stra.update_range(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            stra.update_point(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

        assert stra.get_all_nums() == nums
        return

    def test_segment_tree_range_update_query_sum_min_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeUpdateQuerySumMinMax(len(nums))
        stra.build(nums)

        assert stra.query_min(0, high - 1, low, high - 1, 1) == min(nums)
        assert stra.query_max(0, high - 1, low, high - 1, 1) == max(nums)
        assert stra.query_sum(0, high - 1, low, high - 1, 1) == sum(nums)

        for i in range(high):
            assert stra.query_min(i, i, low, high - 1, 1) == min(nums[i:i + 1])
            assert stra.query_sum(i, i, low, high - 1, 1) == sum(nums[i:i + 1])
            assert stra.query_max(i, i, low, high - 1, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            stra.update_range(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值使用 update
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            stra.update_range(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值使用 update_point
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            stra.update_point(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] += num
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])
        assert stra.get_all_nums() == nums
        return

    def test_segment_tree_range_change_query_sum_min_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        stra = SegmentTreeRangeChangeQuerySumMinMax(nums)

        assert stra.query_min(0, high - 1, low, high - 1, 1) == min(nums)
        assert stra.query_max(0, high - 1, low, high - 1, 1) == max(nums)
        assert stra.query_sum(0, high - 1, low, high - 1, 1) == sum(nums)

        for i in range(high):
            assert stra.query_min(i, i, low, high - 1, 1) == min(nums[i:i + 1])
            assert stra.query_sum(i, i, low, high - 1, 1) == sum(nums[i:i + 1])
            assert stra.query_max(i, i, low, high - 1, 1) == max(nums[i:i + 1])

        for _ in range(high):
            # 区间更新最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            stra.change(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            stra.change(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_max(left, right, low, high - 1, 1) == max(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_max(left, right, low, high - 1, 1) == max(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            # 单点更新最小值 change_point
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            stra.change_point(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_max(left, right, low, high - 1, 1) == max(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_min(left, right, low, high - 1, 1) == min(
                nums[left:right + 1])
            assert stra.query_max(left, right, low, high - 1, 1) == max(
                nums[left:right + 1])
            assert stra.query_sum(left, right, low, high - 1, 1) == sum(
                nums[left:right + 1])

        stra.get_all_nums()
        assert stra.nums == nums
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
        stra = SegmentTreeRangeUpdateSubConSum(nums)

        assert stra.query_max(0, high - 1, low, high - 1, 1)[0] == check(nums)

        for _ in range(high):
            # 区间更新值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            stra.update(left, right, low, high - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert stra.query_max(left, right, low, high - 1, 1)[0] == check(nums[left:right + 1])
        return

    def test_segment_tree_range_change_point_query(self):
        low = 0
        high = 10 ** 9 + 7
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        stra = SegmentTreeRangeUpdateQuery(n)
        for i in range(n):
            stra.update(i, i, 0, n - 1, nums[i], 1)

        for _ in range(10):
            # 区间修改值
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(low, high)
            stra.update(left, right, 0, n - 1, num, 1)
            for i in range(left, right + 1):
                nums[i] = num
            for i in range(n):
                assert stra.query(i, i, 0, n - 1, 1) == nums[i]
        return


if __name__ == '__main__':
    unittest.main()
