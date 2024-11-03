import bisect
import random
import unittest
from itertools import accumulate

from src.structure.tree_array.template import PointAddRangeSum, PointDescendPreMin, RangeAddRangeSum, \
    PointAscendPreMax, PointAscendRangeMax, PointAddRangeSum2D, RangeAddRangeSum2D, \
    PointChangeMaxMin2D, PointXorRangeXor, PointDescendRangeMin, PointChangeRangeSum, PointSetPointAddRangeSum
import math


class TestGeneral(unittest.TestCase):

    def test_point_set_point_add_range_sum(self):

        for _ in range(10):
            ceil = random.randint(10, 1000)
            nums = [random.randint(-ceil, ceil) for _ in range(ceil)]
            tree_array = PointSetPointAddRangeSum(ceil)
            tree_array.build(nums[:])
            for _ in range(ceil):
                d = random.randint(-ceil, ceil)
                i = random.randint(0, ceil - 1)
                x = random.randint(0, 1)
                if x:
                    nums[i] += d
                    tree_array.point_add(i, d)
                else:
                    nums[i] = d
                    tree_array.point_set(i, d)
                assert nums == tree_array.nums
                left = random.randint(0, ceil - 1)
                right = random.randint(left, ceil - 1)
                assert sum(nums[left: right + 1]) == tree_array.range_sum(left, right)
                assert nums == tree_array.get()

        nums = [0, 0, 1, 2, 3, 4, 5]
        pre = list(accumulate(nums))
        tree_array = PointAddRangeSum(len(nums))
        tree_array.build(nums)
        for x in range(sum(nums) + 7):
            assert tree_array.bisect_right(x) == bisect.bisect_right(pre, x)

        nums = [1, 1, 1, 2, 3, 4, 5]
        pre = list(accumulate(nums))
        tree_array = PointAddRangeSum(len(nums))
        tree_array.build(nums)
        for x in range(sum(nums) + 7):
            assert tree_array.bisect_right(x) == bisect.bisect_right(pre, x)
        return

    def test_point_add_range_sum(self):

        for _ in range(10):
            ceil = random.randint(10, 1000)
            nums = [random.randint(-ceil, ceil) for _ in range(ceil)]
            tree_array = PointAddRangeSum(ceil)
            tree_array.build(nums)
            for _ in range(ceil):
                d = random.randint(-ceil, ceil)
                i = random.randint(0, ceil - 1)
                nums[i] += d
                tree_array.point_add(i, d)

                left = random.randint(0, ceil - 1)
                right = random.randint(left, ceil - 1)
                assert sum(nums[left: right + 1]) == tree_array.range_sum(left, right)
                assert nums == tree_array.get()

        nums = [0, 0, 1, 2, 3, 4, 5]
        pre = list(accumulate(nums))
        tree_array = PointAddRangeSum(len(nums))
        tree_array.build(nums)
        for x in range(sum(nums) + 7):
            assert tree_array.bisect_right(x) == bisect.bisect_right(pre, x)

        nums = [1, 1, 1, 2, 3, 4, 5]
        pre = list(accumulate(nums))
        tree_array = PointAddRangeSum(len(nums))
        tree_array.build(nums)
        for x in range(sum(nums) + 7):
            assert tree_array.bisect_right(x) == bisect.bisect_right(pre, x)
        return

    def test_point_change_range_sum(self):

        for _ in range(10):
            ceil = random.randint(10, 1000)
            nums = [random.randint(-ceil, ceil) for _ in range(ceil)]
            tree_array = PointChangeRangeSum(ceil)
            tree_array.build(nums)
            for _ in range(ceil):
                d = random.randint(-ceil, ceil)
                i = random.randint(0, ceil - 1)
                nums[i] = d
                tree_array.point_change(i + 1, d)

                left = random.randint(0, ceil - 1)
                right = random.randint(left, ceil - 1)
                assert sum(nums[left: right + 1]) == tree_array.range_sum(left + 1, right + 1)
                assert nums == tree_array.get()
        return

    def test_point_ascend_pre_max(self):
        for initial in [-math.inf, 0]:
            for _ in range(10):
                n = random.randint(10, 1000)
                low = -1000 if initial == -math.inf else 0
                high = 10000
                tree_array = PointAscendPreMax(n, initial)
                nums = [initial] * n
                for _ in range(100):
                    x = random.randint(low, high)
                    i = random.randint(0, n - 1)
                    nums[i] = nums[i] if nums[i] > x else x
                    tree_array.point_ascend(i + 1, x)
                    assert list(accumulate(nums, max)) == [tree_array.pre_max(i) for i in range(n)]
        return

    def test_point_ascend_range_max(self):
        for initial in [-math.inf, 0]:
            for _ in range(10):
                n = random.randint(10, 1000)
                low = -1000 if initial == -math.inf else 0
                high = 10000
                tree_array = PointAscendRangeMax(n, initial)
                nums = [initial] * n
                for _ in range(100):
                    x = random.randint(low, high)
                    i = random.randint(0, n - 1)
                    nums[i] = nums[i] if nums[i] > x else x
                    tree_array.point_ascend(i + 1, x)
                    ll = random.randint(0, n - 1)
                    rr = random.randint(ll, n - 1)
                    assert max(nums[ll:rr + 1]) == tree_array.range_max(ll + 1, rr + 1)
        return

    def test_point_descend_pre_min(self):
        for initial in [math.inf, 10000]:
            for _ in range(10):
                n = random.randint(10, 1000)
                low = -10000
                high = 10000
                tree_array = PointDescendPreMin(n, initial)
                nums = [initial] * n
                for _ in range(100):
                    x = random.randint(low, high)
                    i = random.randint(0, n - 1)
                    nums[i] = nums[i] if nums[i] < x else x
                    tree_array.point_descend(i + 1, x)
                    assert list(accumulate(nums, min)) == [tree_array.pre_min(i) for i in range(n)]
        return

    def test_point_descend_range_min(self):
        for initial in [math.inf, 10000]:
            for _ in range(10):
                n = random.randint(10, 1000)
                low = -10000
                high = 10000
                tree_array = PointDescendRangeMin(n, initial)
                nums = [initial] * n
                for _ in range(100):
                    x = random.randint(low, high)
                    i = random.randint(0, n - 1)
                    nums[i] = nums[i] if nums[i] < x else x
                    tree_array.point_descend(i + 1, x)
                    ll = random.randint(0, n - 1)
                    rr = random.randint(ll, n - 1)
                    assert min(nums[ll:rr + 1]) == tree_array.range_min(ll + 1, rr + 1)
        return

    def test_range_add_range_sum(self):

        for _ in range(10):
            for _ in range(10):
                n = random.randint(10, 1000)
                nums = [random.randint(-10000, 10000) for _ in range(n)]
                tree_array = RangeAddRangeSum(n)
                tree_array.build(nums)
                for _ in range(10):
                    x = random.randint(-10000, 10000)
                    ll = random.randint(0, n - 1)
                    rr = random.randint(ll, n - 1)
                    for j in range(ll, rr + 1):
                        nums[j] += x
                    tree_array.range_add(ll + 1, rr + 1, x)
                    assert tree_array.get() == nums
                    ll = random.randint(0, n - 1)
                    rr = random.randint(ll, n - 1)
                    assert tree_array.range_sum(ll + 1, rr + 1) == sum(nums[ll: rr + 1])
        return

    def test_point_add_range_sum_2d(self):

        # tree_matrix|，单点增减，区间查询
        m = n = 100
        high = 100000
        tree = PointAddRangeSum2D(m, n)
        grid = [[random.randint(-high, high) for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                tree.point_add(i + 1, j + 1, grid[i][j])
        for _ in range(m):
            row = random.randint(0, m - 1)
            col = random.randint(0, n - 1)
            x = random.randint(-high, high)
            grid[row][col] += x
            tree.point_add(row + 1, col + 1, x)
            x1 = random.randint(0, m - 1)
            y1 = random.randint(0, n - 1)
            x2 = random.randint(x1, m - 1)
            y2 = random.randint(y1, n - 1)
            assert tree.range_sum(x1 + 1, y1 + 1, x2 + 1, y2 + 1) == sum(sum(g[y1:y2 + 1]) for g in grid[x1:x2 + 1])
        return

    def test_range_add_range_sum_2d(self):
        # tree_matrix|，区间增减，区间查询
        m = n = 100
        high = 100000
        tree = RangeAddRangeSum2D(m, n)
        grid = [[random.randint(-high, high) for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                tree.range_add(i + 1, j + 1, i + 1, j + 1, grid[i][j])
        for _ in range(m):
            x1 = random.randint(0, m - 1)
            y1 = random.randint(0, n - 1)
            x2 = random.randint(x1, m - 1)
            y2 = random.randint(y1, n - 1)
            x = random.randint(-high, high)
            for i in range(x1, x2 + 1):
                for j in range(y1, y2 + 1):
                    grid[i][j] += x
            tree.range_add(x1 + 1, y1 + 1, x2 + 1, y2 + 1, x)
            x1 = random.randint(0, m - 1)
            y1 = random.randint(0, n - 1)
            x2 = random.randint(x1, m - 1)
            y2 = random.randint(y1, n - 1)
            assert tree.range_query(x1 + 1, y1 + 1, x2 + 1, y2 + 1) == sum(
                sum(g[y1:y2 + 1]) for g in grid[x1:x2 + 1])
        return

    @unittest.skip
    def test_point_change_max_min_2d(self):

        # tree_matrix|，单点增减，区间查询
        random.seed(2023)
        m = n = 100
        high = 100000
        tree = PointChangeMaxMin2D(m, n)
        grid = [[random.randint(0, high) for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                tree.add(i + 1, j + 1, grid[i][j])
        for _ in range(m):
            row = random.randint(0, m - 1)
            col = random.randint(0, n - 1)
            x = random.randint(0, high)
            grid[row][col] += x
            tree.add(row + 1, col + 1, grid[row][col])
            x1 = random.randint(0, m - 1)
            y1 = random.randint(0, n - 1)
            x2 = random.randint(x1, m - 1)
            y2 = random.randint(y1, n - 1)
            ans1 = tree.find_max(x1 + 1, y1 + 1, x2 + 1, y2 + 1)
            ans2 = max(max(g[y1:y2 + 1]) for g in grid[x1:x2 + 1])
            print(ans1, ans2)
            assert ans1 == ans2
        return

    def test_point_xor_range_xor(self):
        for _ in range(10):
            n = random.randint(10, 1000)
            low = -10000
            high = 10000
            nums = [random.randint(low, high) for _ in range(n)]
            tree_array = PointXorRangeXor(n)
            tree_array.build(nums)
            for _ in range(100):
                x = random.randint(low, high)
                i = random.randint(0, n - 1)
                nums[i] ^= x
                tree_array.point_xor(i + 1, x)
                assert tree_array.get() == nums
                ll = random.randint(0, n - 1)
                rr = random.randint(ll, n - 1)
                res = 0
                for num in nums[ll: rr + 1]:
                    res ^= num
                assert res == tree_array.range_xor(ll + 1, rr + 1)
        return


if __name__ == '__main__':
    unittest.main()
