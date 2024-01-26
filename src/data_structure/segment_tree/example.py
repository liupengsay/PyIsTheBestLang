import random
import unittest
from functools import reduce
from operator import or_, xor

from src.data_structure.segment_tree.template import RangeAscendRangeMax, \
    RangeDescendRangeMin, \
    RangeAddRangeSumMinMax, RangeSetRangeSumMinMax, RangeSetRangeMaxNonEmpConSubSum, \
    RangeOrRangeAnd, \
    RangeSetRangeSumMinMaxDynamic, RangeSetRangeOr, RangeAffineRangeSum, RangeAddMulRangeSum, \
    RangeSetAddRangeSumMinMax, RangeXorUpdateRangeXorQuery, RangeSetReverseRangeSumLongestConSub, PointSetRangeMinCount, \
    RangeAddPointGet


class TestGeneral(unittest.TestCase):

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
                tree.range_set_add(left, right, (-tree.inf, num))
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
        high = 10000
        n = 10000
        nums = [0] * n
        tree = RangeSetRangeSumMinMaxDynamic(n)

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


if __name__ == '__main__':
    unittest.main()
