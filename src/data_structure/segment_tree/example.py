import random
import unittest
from functools import reduce

from operator import or_, xor

from src.data_structure.segment_tree.template import RangeAscendRangeMax, \
    RangeDescendRangeMin, RangeAscendRangeMaxIndex, \
    RangeAddRangeSumMinMax, RangeSetRangeSumMinMax, RangeSetRangeMaxNonEmpConSubSum, \
    RangeOrRangeAnd, \
    RangeSetRangeSumMinMaxDynamic, RangeSetRangeOr, RangeAffineRangeSum, RangeAddMulRangeSum, \
    RangeSetAddRangeSumMinMax, RangeXorUpdateRangeXorQuery, RangeSetReverseRangeSumLongestConSub, PointSetRangeMinCount, \
    RangeAddPointGet, RangeSetRangeSegCountLength, RangeAddRangeWeightedSum, \
    RangeChminChmaxPointGet, RangeSetPreSumMaxDynamic, RangeSetPreSumMaxDynamicDct, RangeSetRangeSumMinMaxDynamicDct, \
    RangeRevereRangeAlter, RangeAddRangeMinCount, RangeSetPointGet, PointSetPreMinPostMin, PointSetPreMaxPostMin, \
    RangeAddRangeMaxGainMinGain, SegmentTreeOptBuildGraph, SegmentTreeOptBuildGraphZKW


class TestGeneral(unittest.TestCase):

    def test_segment_tree_build_graph(self):
        """
        example： 1
                / \
               /   \
              2     3
             / \   / \
            4   5 6   7
        """
        n = 8
        tree = SegmentTreeOptBuildGraph(n)
        assert tree.edges == [(1, 2), (1, 3), (2, 4), (2, 5), (4, 8), (4, 9), (5, 10), (5, 11), (3, 6), (3, 7), (6, 12),
                              (6, 13), (7, 14), (7, 15)]
        assert tree.range_opt(2, 5) == [5, 6]
        assert tree.range_opt(0, 7) == [1]
        assert tree.range_opt(3, 7) == [11, 3]
        assert tree.range_opt(1, 2) == [9, 10]
        assert tree.range_opt(1,1) == [9]
        assert tree.range_opt(1, 4) == [9, 5, 12]
        for _ in range(100):
            n = random.randint(1, 10 ** 3)
            tree = SegmentTreeOptBuildGraph(n)
            assert len(tree.edges) == 2 * (n - 1)
            nodes = []
            for edge in tree.edges:
                nodes.extend(edge)
            assert len(set(nodes)) <= 2*n
            assert n==1 or max(nodes) <= 4*n
            assert len(tree.leaves) == n
        return

    def test_segment_tree_build_graph_zkw(self):
        """
        example： 1
                / \
               /   \
              2     3
             / \   / \
            4   5 6   7
        """
        n = 8
        tree = SegmentTreeOptBuildGraphZKW(n, True)
        assert sorted(tree.edges) == sorted([(1, 2), (1, 3), (2, 4), (2, 5), (4, 8), (4, 9), (5, 10), (5, 11), (3, 6), (3, 7), (6, 12),
                              (6, 13), (7, 14), (7, 15)])
        assert tree.range_opt(2, 5) == [5, 6]
        assert tree.range_opt(0, 7) == [1]
        assert tree.range_opt(3, 7) == [11, 3]
        assert tree.range_opt(1, 2) == [9, 10]
        assert tree.range_opt(1,1) == [9]
        assert sorted(tree.range_opt(1, 4)) == sorted([9, 5, 12])
        for _ in range(100):
            n = random.randint(1, 10 ** 3)
            tree = SegmentTreeOptBuildGraphZKW(n, True)
            assert len(tree.edges) == 2 * (n - 1)
            nodes = []
            for edge in tree.edges:
                nodes.extend(edge)
            assert len(set(nodes)) <= 2*n
            assert n==1 or max(nodes) <= 2*n
        return

    def test_max_gain_min_gain(self):
        for _ in range(1000):
            n = 1000
            tree = RangeAddRangeMaxGainMinGain(n)
            nums = [random.randint(0, n) for _ in range(n)]
            tree.build(nums)
            for _ in range(100):
                ll = random.randint(0, n - 2)
                rr = random.randint(ll + 1, n - 1)
                ans = tree.range_max_gain_min_gain(ll, rr)
                assert ans[0] == min(nums[ll:rr + 1])
                assert ans[1] == max(nums[ll:rr + 1])
                floor = nums[ll]
                ceil = nums[ll]
                cover1 = -inf
                cover2 = inf
                for x in range(ll + 1, rr + 1):
                    cover1 = max(cover1, nums[x] - floor)
                    cover2 = min(cover2, nums[x] - ceil)
                    ceil = max(ceil, nums[x])
                    floor = min(floor, nums[x])
                assert ans[2] == cover1
                assert ans[3] == cover2
                v = random.randint(0, 100)
                tree.range_add(ll, rr, v)
                for i in range(ll, rr + 1):
                    nums[i] += v
        return

    def test_point_set_pre_min_post_min(self):
        for _ in range(1000):
            n = 1000
            tree = PointSetPreMinPostMin(n)
            nums = [random.randint(0, n) for _ in range(n)]
            tree.build(nums)
            for _ in range(10):
                ind = random.randint(0, n - 1)
                v = random.randint(0, n)
                nums[ind] = v
                tree.point_set(ind, v)
                assert tree.get() == nums
                ind = random.randint(0, n - 1)
                assert tree.pre_min(ind) == min(nums[:ind + 1])

                assert tree.post_min(ind) == min(nums[ind:])
        return

    def test_point_set_pre_max_post_min(self):
        for _ in range(1000):
            n = 1000
            tree = PointSetPreMaxPostMin(n)
            nums = [random.randint(0, n) for _ in range(n)]
            tree.build(nums)
            for _ in range(10):
                ind = random.randint(0, n - 1)
                v = random.randint(0, n)
                nums[ind] = v
                tree.point_set(ind, v)
                assert tree.get() == nums
                ind = random.randint(0, n - 1)
                assert tree.pre_max(ind) == max(nums[:ind + 1])

                assert tree.post_min(ind) == min(nums[ind:])
        return

    def test_range_set_point_get(self):
        for _ in range(1000):
            n = 1000
            tree = RangeSetPointGet(n)
            nums = [random.randint(0, 1) for _ in range(n)]
            tree.build(nums)
            for _ in range(10):
                ll = random.randint(0, n - 1)
                rr = random.randint(ll, n - 1)
                v = random.randint(0, 1)
                for i in range(ll, rr + 1):
                    nums[i] = v
                tree.range_set(ll, rr, v)
                assert tree.get() == nums
        return

    def test_range_reverse_alter_query(self):
        for _ in range(100):
            n = 100
            tree = RangeRevereRangeAlter(n)
            nums = [random.randint(0, 1) for _ in range(n)]
            tree.build(nums)
            for _ in range(10):
                ll = random.randint(0, n - 1)
                rr = random.randint(ll, n - 1)
                for i in range(ll, rr + 1):
                    nums[i] ^= 1
                tree.range_reverse(ll, rr)
                assert tree.get() == nums
                ll = random.randint(0, n - 1)
                rr = random.randint(ll, n - 1)
                res = tree.range_alter_query(ll, rr)
                check = all(nums[i + 1] != nums[i] for i in range(ll, rr))
                assert res == check
        return

    def test_range_add_point_get(self):
        low = 1
        high = 100
        n = 100
        nums = [random.randint(low, high) for _ in range(n)]
        tree = RangeAddPointGet(n)
        tree.build(nums)
        for _ in range(10000):
            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            num = random.randint(low, high)
            for i in range(ll, rr + 1):
                nums[i] += num
            tree.range_add(ll, rr, num)
        assert nums == tree.get() == [tree.point_get(i) for i in range(n)]
        return

    def test_point_set_range_min_count(self):
        low = 1
        high = 100
        n = 100
        nums = [random.randint(low, high) for _ in range(n)]
        tree = PointSetRangeMinCount(n, 0)
        tree.build(nums)
        for _ in range(10000):
            i = random.randint(0, n - 1)
            num = random.randint(low, high)
            nums[i] = num
            tree.point_set(i, num)

            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            low = min(nums[ll:rr + 1])
            cnt = nums[ll:rr + 1].count(low)
            res = tree.range_min_count(ll, rr)
            assert res == (low, cnt)
        assert nums == tree.get()
        return

    def test_range_add_range_min_count(self):
        low = 1
        high = 100
        n = 1000
        nums = [random.randint(low, high) for _ in range(n)]
        tree = RangeAddRangeMinCount(n)
        tree.build(nums)
        for _ in range(10000):
            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            num = random.randint(-high, high)
            for i in range(ll, rr + 1):
                nums[i] += num
            tree.range_add(ll, rr, num)

            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            floor = min(nums[ll:rr + 1])
            cnt = nums[ll:rr + 1].count(floor)
            res = tree.range_min_count(ll, rr)
            assert res == (floor, cnt)
        assert nums == tree.get()
        return

    def test_range_or_range_and(self):
        low = 0
        high = (1 << 31) - 1
        n = 1000
        nums = [random.randint(low, high) for _ in range(n)]
        tree = RangeOrRangeAnd(n)
        tree.build(nums)
        for _ in range(1000):
            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            num = random.randint(low, high)
            for i in range(ll, rr + 1):
                nums[i] |= num
            tree.range_or(ll, rr, num)

            ll = random.randint(0, n - 1)
            rr = random.randint(ll, n - 1)
            res = high
            for i in range(ll, rr + 1):
                res &= nums[i]
            assert res == tree.range_and(ll, rr)
        assert nums == tree.get()
        return

    def test_range_ascend_range_max(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        tree = RangeAscendRangeMax(high)
        tree.build(nums)
        assert tree.get() == nums
        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            tree.range_ascend(left, right, num)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_max(left, right) == max(nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            tree.range_ascend(left, right, num)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num
            assert tree.range_max(left, right) == max(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_max(left, right) == max(nums[left:right + 1])
        assert tree.get() == nums
        return

    def test_range_ascend_range_max_index(self):
        random.seed(2024)
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        tree = RangeAscendRangeMaxIndex(high)
        tree.build(nums)
        assert tree.get() == nums
        for _ in range(high):
            # 区间更新最大值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            tree.range_ascend(left, right, 0, num)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            res = tree.range_max_index(left, right)
            assert res[0] == max(nums[left:right + 1])

            # 单点更新最大值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            tree.range_ascend(left, right, 0, num)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] > num else num
            res = tree.range_max_index(left, right)
            assert res[0] == max(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            res = tree.range_max_index(left, right)
            assert res[0] == max(nums[left:right + 1])
        assert tree.get() == nums
        return

    def test_range_descend_range_min(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        tree = RangeDescendRangeMin(high)
        tree.build(nums)

        for _ in range(high):
            # 区间更新与查询最小值
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            tree.range_descend(left, right, num)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_min(left, right) == min(nums[left:right + 1])

            # 单点更新与查询最小值
            left = random.randint(0, high - 1)
            right = left
            num = random.randint(low, high)
            tree.range_descend(left, right, num)
            for i in range(left, right + 1):
                nums[i] = nums[i] if nums[i] < num else num
            assert tree.range_min(left, right) == min(nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_min(left, right) == min(nums[left:right + 1])

        assert tree.get() == nums
        return

    def test_range_add_range_sum_min_max(self):
        low = -10000
        high = 10000
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

    def test_range_chmin_chmax_add_range_sum_min_max(self):
        low = -2000
        high = 2000
        nums = [random.randint(low, high) for _ in range(high)]
        tree = RangeChminChmaxPointGet(high, -high, high)
        tree.build(nums)
        assert tree.get() == nums

        for _ in range(high):

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            tree.range_chmin_chmax(left, right, num, tree.high_initial)
            for i in range(left, right + 1):
                nums[i] = max(num, nums[i])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            tree.range_chmin_chmax(left, right, tree.low_initial, num)
            for i in range(left, right + 1):
                nums[i] = min(num, nums[i])
            assert tree.get() == nums
            assert [tree.point_get(i) for i in range(high)] == nums
        return

    def test_range_set_reverse_range_sum_longest_con_sub_sum(self):
        random.seed(2023)
        high = 10000
        nums = [random.randint(0, 1) for _ in range(high)]
        tree = RangeSetReverseRangeSumLongestConSub(high)
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

        assert tree.range_sum(0, high - 1) == sum(nums)
        assert tree.get() == nums
        for _ in range(100):
            op = random.randint(0, 4)
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.get() == nums
            if op == 0:
                tree.range_set_reverse(left, right, op)
                for i in range(left, right + 1):
                    nums[i] = 0
            elif op == 1:
                tree.range_set_reverse(left, right, op)
                for i in range(left, right + 1):
                    nums[i] = 1
            elif op == 2:
                tree.range_set_reverse(left, right, op)
                for i in range(left, right + 1):
                    nums[i] = 1 - nums[i]
            elif op == 3:
                assert tree.range_sum(left, right) == sum(nums[left:right + 1])
            else:
                assert tree.range_longest_con_sub(left, right) == check(nums[left:right + 1])
        assert tree.get() == nums
        return

    def test_range_xor_update_range_xor_query(self):
        low = 0
        high = 10000
        nums = [random.randint(low, high) for _ in range(high)]
        tree = RangeXorUpdateRangeXorQuery(high)
        tree.build(nums)

        assert tree.range_xor_query(0, high - 1) == reduce(xor, nums)

        for _ in range(high):

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(0, high)
            tree.range_xor_update(left, right, num)
            for i in range(left, right + 1):
                nums[i] ^= num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)

            assert tree.range_xor_query(left, right) == reduce(xor, nums[left:right + 1])

        assert tree.get() == nums
        return

    def test_range_set_add_range_max(self):
        random.seed(2024)
        low = -10 ** 9
        high = 10 ** 9

        for _ in range(100):
            n = random.randint(1000, 10 ** 4)
            nums = [random.randint(low, high) for _ in range(n)]
            tree = RangeSetAddRangeSumMinMax(n)
            tree.build(nums)
            assert tree.range_max(0, high - 1) == max(nums)

            for _ in range(100):

                left = random.randint(0, n - 1)
                right = random.randint(left, n - 1)
                num = random.randint(low, high)
                tree.range_set_add(left, right, (-tree.initial, num))
                for i in range(left, right + 1):
                    nums[i] += num
                left = random.randint(0, n - 1)
                right = random.randint(left, n - 1)
                assert tree.range_max(left, right) == max(nums[left:right + 1])
                assert tree.range_min(left, right) == min(nums[left:right + 1])
                assert tree.range_sum(left, right) == sum(nums[left:right + 1])

                left = random.randint(0, n - 1)
                right = random.randint(left, n - 1)
                num = random.randint(low, high)
                tree.range_set_add(left, right, (num, 0))
                for i in range(left, right + 1):
                    nums[i] = num
                left = random.randint(0, n - 1)
                right = random.randint(left, n - 1)
                assert tree.range_max(left, right) == max(nums[left:right + 1])
                assert tree.range_min(left, right) == min(nums[left:right + 1])
                assert tree.range_sum(left, right) == sum(nums[left:right + 1])

            assert tree.get() == nums
        return

    def test_range_add_mul_range_sum(self):
        low = -10000
        high = 10000
        mod = 10 ** 9 + 7
        nums = [random.randint(low, high) % mod for _ in range(high)]
        tree = RangeAddMulRangeSum(high, mod)
        tree.build(nums)

        assert tree.range_sum(0, high - 1) == sum(nums) % mod

        for _ in range(high):

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            tree.range_add_mul(left, right, num, "add")
            for i in range(left, right + 1):
                nums[i] += num
                nums[i] %= mod
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1]) % mod

            left = random.randint(0, high - 1)
            right = left
            num = random.randint(-high, high)
            tree.range_add_mul(left, right, num, "mul")
            for i in range(left, right + 1):
                nums[i] *= num
                nums[i] %= mod
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1]) % mod

        ans = [tree.range_sum(i, i) for i in range(high)]
        assert tree.get() == nums == ans
        return

    def test_range_affine_range_sum(self):
        low = -10000
        high = 10000
        mod = 10 ** 9 + 7
        nums = [random.randint(low, high) % mod for _ in range(high)]
        tree = RangeAffineRangeSum(high, mod)
        tree.build(nums)

        assert tree.range_sum(0, high - 1) == sum(nums) % mod

        for _ in range(high):

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(1, high)
            tree.range_affine(left, right, (1 << 32) | num)
            for i in range(left, right + 1):
                nums[i] += num
                nums[i] %= mod
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1]) % mod

            left = random.randint(0, high - 1)
            right = left
            num = random.randint(1, high)
            tree.range_affine(left, right, num << 32)
            for i in range(left, right + 1):
                nums[i] *= num
                nums[i] %= mod
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1]) % mod

        ans = [tree.range_sum(i, i) for i in range(high)]
        assert tree.get() == nums == ans
        return

    def test_range_set_range_sum_min_max(self):
        low = -10000
        high = 10000
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        tree = RangeSetRangeSumMinMax(n)
        tree.build(nums)
        assert tree.range_min(0, n - 1) == min(nums)
        assert tree.range_max(0, n - 1) == max(nums)
        assert tree.range_sum(0, n - 1) == sum(nums)
        assert tree.get() == nums

        for _ in range(high):
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
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
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

        assert tree.get() == nums
        return

    def test_range_set_range_or(self):
        low = 0
        high = 10000
        n = 10000
        nums = [random.randint(low, high) for _ in range(n)]
        tree = RangeSetRangeOr(n)
        tree.build(nums)

        for _ in range(high):
            # 区间修改
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(0, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_or(left, right) == reduce(or_, nums[left:right + 1])

        assert tree.get() == nums
        return

    def test_range_set_range_sum_min_max_dynamic(self):
        high = n = 10000
        nums = [0] * n
        tree = RangeSetRangeSumMinMaxDynamic(n, -high - 1)

        for _ in range(high):
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
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
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

        assert tree.range_min(0, n - 1) == min(nums)
        assert tree.range_max(0, n - 1) == max(nums)
        assert tree.range_sum(0, n - 1) == sum(nums)

        low = 0
        high = 10
        n = 1000
        nums = [0] * n
        tree = RangeSetRangeSumMinMaxDynamic(n, -1)

        for _ in range(high):
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
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
            num = random.randint(low, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

            cur = random.randint(0, sum(nums) + 1)
            ans = 0
            pre = cur
            for num in nums:
                if cur >= num:
                    ans += 1
                    cur -= num
                else:
                    break
            assert ans == tree.range_sum_bisect_left(pre)
            for cur in [0, sum(nums), sum(nums) + 1]:
                ans = 0
                pre = cur
                for num in nums:
                    if cur >= num:
                        ans += 1
                        cur -= num
                    else:
                        break
                assert ans == tree.range_sum_bisect_left(pre)
        return

    def test_range_set_range_sum_min_max_dynamic_dct(self):
        high = 10000
        n = 10000
        nums = [0] * n
        tree = RangeSetRangeSumMinMaxDynamicDct(n, 30, -high - 1)

        for _ in range(high):
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
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
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

        assert tree.range_min(0, n - 1) == min(nums)
        assert tree.range_max(0, n - 1) == max(nums)
        assert tree.range_sum(0, n - 1) == sum(nums)

        low = 0
        high = 10
        n = 1000
        nums = [0] * n
        tree = RangeSetRangeSumMinMaxDynamic(n, -1)

        for _ in range(high):
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(low, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
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
            num = random.randint(low, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.range_min(left, right) == min(
                nums[left:right + 1])
            assert tree.range_max(left, right) == max(
                nums[left:right + 1])
            assert tree.range_sum(left, right) == sum(
                nums[left:right + 1])

            cur = random.randint(0, sum(nums) + 1)
            ans = 0
            pre = cur
            for num in nums:
                if cur >= num:
                    ans += 1
                    cur -= num
                else:
                    break
            assert ans == tree.range_sum_bisect_left(pre)
            for cur in [0, sum(nums), sum(nums) + 1]:
                ans = 0
                pre = cur
                for num in nums:
                    if cur >= num:
                        ans += 1
                        cur -= num
                    else:
                        break
                assert ans == tree.range_sum_bisect_left(pre)
        return

    def test_range_set_pre_sum_max_dynamic(self):
        high = n = 1000
        nums = [0] * n
        tree = RangeSetPreSumMaxDynamic(n, -high - 1)

        for _ in range(high):
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(-high, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            pre = [0]
            for num in nums:
                pre.append(pre[-1] + num)
            assert tree.range_sum(left, right) == sum(nums[left:right + 1])
            assert tree.pre_sum_max[1] == max(pre[1:])
            assert tree.range_pre_sum_max(n - 1) == max(pre[1:])
            for i in range(1, len(pre)):
                assert max(pre[1:i + 1]) == tree.range_pre_sum_max(i - 1)
            cur = random.randint(0, max(pre) + 10)
            ans = pre = 0
            for num in nums:
                if pre + num > cur:
                    break
                else:
                    ans += 1
                    pre += num
            assert ans == tree.range_pre_sum_max_bisect_left(cur)
        return

    def test_range_set_pre_sum_max_dynamic_dct(self):
        high = n = 1000
        nums = [0] * n
        tree = RangeSetPreSumMaxDynamicDct(n, 3 * 10, -high - 1)

        for _ in range(high):
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            num = random.randint(-high, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, n - 1)
            right = random.randint(left, n - 1)
            pre = [0]
            for num in nums:
                pre.append(pre[-1] + num)
            assert tree.range_sum(left, right) == sum(nums[left:right + 1])
            assert tree.pre_sum_max[1] == max(pre[1:])
            assert tree.range_pre_sum_max(n - 1) == max(pre[1:])
            for i in range(1, len(pre)):
                assert max(pre[1:i + 1]) == tree.range_pre_sum_max(i - 1)
            cur = random.randint(0, max(pre) + 10)
            ans = pre = 0
            for num in nums:
                if pre + num > cur:
                    break
                else:
                    ans += 1
                    pre += num
            assert ans == tree.range_pre_sum_max_bisect_left(cur)
        return

    def test_range_set_range_max_non_emp_son_sub_sum(self):

        def check(lst):
            pre = ans = lst[0]
            for x in lst[1:]:
                pre = pre + x if pre + x > x else x
                ans = ans if ans > pre else pre
            return ans

        low = -10
        high = 10
        nums = [random.randint(low, high) for _ in range(high)]
        tree = RangeSetRangeMaxNonEmpConSubSum(high, high + 1)
        tree.build(nums)
        assert tree.range_max_non_emp_con_sub_sum(0, high - 1) == check(nums)
        assert tree.get() == nums
        for _ in range(200):
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            num = random.randint(-high, high)
            tree.range_set(left, right, num)
            for i in range(left, right + 1):
                nums[i] = num
            left = random.randint(0, high - 1)
            right = random.randint(left, high - 1)
            assert tree.get() == nums
            assert tree.range_max_non_emp_con_sub_sum(left, right) == check(nums[left:right + 1])
        return

    def test_range_set_range_seg_count_length(self):

        def check(lst):
            cnt = s = 0
            pre = lst[0]
            c = 1
            for a in lst[1:]:
                if a == pre:
                    c += 1
                else:
                    if pre:
                        cnt += 1
                        s += c
                    pre = a
                    c = 1
            if pre:
                cnt += 1
                s += c
            return cnt, s

        low = 0
        high = 1
        for x in range(5):
            n = 10 ** x
            nums = [random.randint(low, high) for _ in range(n)]
            tree = RangeSetRangeSegCountLength(n, -1)
            tree.build(nums)
            assert tree.range_seg_count_length(0, n - 1) == check(nums)
            assert tree.get() == nums
            for _ in range(1000):
                left = random.randint(0, n - 1)
                right = random.randint(left, n - 1)
                num = random.randint(low, high)
                tree.range_set(left, right, num)
                for i in range(left, right + 1):
                    nums[i] = num
                assert (tree.cover[1], tree.sum[1]) == check(nums)

                left = random.randint(0, n - 1)
                right = random.randint(left, n - 1)
                assert tree.get() == nums
                assert tree.range_seg_count_length(left, right) == check(nums[left:right + 1])
                assert tree.range_seg_count_length(0, n - 1) == check(nums)
                assert tree.left[1] == nums[0]
                assert tree.right[1] == nums[-1]
                assert tree.sum[1] == sum(nums)
                assert tree.cover[1] == check(nums)[0]

        return

    def test_range_add_range_weighted_sum(self):

        def check(lst):
            res = sum(xx * (ind + 1) for ind, xx in enumerate(lst))
            return res

        low = -10
        high = 100
        for x in range(5):
            n = 10 ** x
            nums = [random.randint(low, high) for _ in range(n)]
            tree = RangeAddRangeWeightedSum(n)
            tree.build(nums)
            assert tree.get() == nums
            for _ in range(1000):
                left = random.randint(0, n - 1)
                right = random.randint(left, n - 1)
                num = random.randint(low, high)
                tree.range_add(left, right, num)
                for i in range(left, right + 1):
                    nums[i] += num
                left = random.randint(0, n - 1)
                right = random.randint(left, n - 1)
                assert tree.range_weighted_sum(left, right) == check(nums[left:right + 1])
            assert tree.get() == nums
        return


if __name__ == '__main__':
    unittest.main()
