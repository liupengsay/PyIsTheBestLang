import math
import random
import unittest
from functools import reduce

from src.data_structure.sparse_table.template import SparseTable2D, SparseTable1, SparseTable2


class TestGeneral(unittest.TestCase):

    def test_sparse_table(self):

        def check_and(lst):
            ans = lst[0]
            for num in lst[1:]:
                ans &= num
            return ans

        def check_or(lst):
            ans = lst[0]
            for num in lst[1:]:
                ans |= num
            return ans

        nums = [9, 3, 1, 7, 5, 6, 0, 8]
        st = SparseTable1(nums)
        queries = [[1, 6], [1, 5], [2, 7], [2, 6], [1, 8], [4, 8], [3, 7], [1, 8]]
        assert [st.query(left, right) for left, right in queries] == [9, 9, 7, 7, 9, 8, 7, 9]

        ceil = 1000
        nums = [random.randint(0, ceil) for _ in range(1000)]
        st1_max = SparseTable1(nums, "max")
        st1_min = SparseTable1(nums, "min")
        st1_gcd = SparseTable1(nums, "gcd")
        st1_lcm = SparseTable1(nums, "lcm")
        st1_and = SparseTable1(nums, "and")
        st1_or = SparseTable1(nums, "or")

        st2_max = SparseTable2(nums, "max")
        st2_min = SparseTable2(nums, "min")
        st2_gcd = SparseTable2(nums, "gcd")
        for _ in range(ceil):
            left = random.randint(1, ceil - 10)
            right = random.randint(left, ceil)
            assert st1_max.query(left, right) == st2_max.query(left - 1, right - 1) == max(nums[left - 1:right])
            assert st1_min.query(left, right) == st2_min.query(left - 1, right - 1) == min(nums[left - 1:right])
            assert st1_gcd.query(left, right) == st2_gcd.query(left - 1, right - 1) == reduce(math.gcd,
                                                                                              nums[left - 1:right])
            assert st1_lcm.query(left, right) == reduce(math.lcm, nums[left - 1:right])
            assert st1_and.query(left, right) == check_and(nums[left - 1:right])
            assert st1_or.query(left, right) == check_or(nums[left - 1:right])
        return

    def test_sparse_table_2d_max_min(self):

        # 二维稀疏表
        m = n = 50
        high = 100000
        grid = [[random.randint(0, high) for _ in range(n)] for _ in range(m)]

        for method in ["max", "min", "lcm", "gcd", "or", "and"]:
            st = SparseTable2D(grid, method)
            x1 = random.randint(0, m - 1)
            y1 = random.randint(0, n - 1)
            x2 = random.randint(x1, m - 1)
            y2 = random.randint(y1, n - 1)

            ans1 = st.query(x1, y1, x2, y2)
            ans2 = st.fun([st.fun(g[y1:y2 + 1]) for g in grid[x1:x2 + 1]])
            assert ans1 == ans2
        return


if __name__ == '__main__':
    unittest.main()
