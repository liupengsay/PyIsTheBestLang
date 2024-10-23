import math
import random
import unittest
from functools import reduce
from itertools import accumulate
from operator import or_, and_

from src.structure.sparse_table.template import SparseTable2D, SparseTable


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
        st = SparseTable(nums, max)
        queries = [[1, 6], [1, 5], [2, 7], [2, 6], [1, 8], [4, 8], [3, 7], [1, 8]]
        assert [st.query(left - 1, right - 1) for left, right in queries] == [9, 9, 7, 7, 9, 8, 7, 9]

        ceil = 2000
        nums = [random.randint(1, ceil) for _ in range(2000)]
        st1_max = SparseTable(nums, max)
        st1_min = SparseTable(nums, min)
        st1_gcd = SparseTable(nums, math.gcd)
        st1_lcm = SparseTable(nums, math.lcm)
        st1_and = SparseTable(nums, and_)
        st1_or = SparseTable(nums, or_)

        for _ in range(ceil):
            left = random.randint(1, ceil - 10)
            right = random.randint(left, ceil)
            left -= 1
            right -= 1
            assert st1_max.query(left, right) == max(nums[left:right + 1])
            assert st1_min.query(left, right) == min(nums[left:right + 1])
            assert st1_gcd.query(left, right) == reduce(math.gcd, nums[left:right + 1])
            assert st1_lcm.query(left, right) == reduce(math.lcm, nums[left:right + 1])
            assert st1_and.query(left, right) == check_and(nums[left:right + 1])
            assert st1_or.query(left, right) == check_or(nums[left:right + 1])
            pre = list(accumulate(nums[left:], and_))
            for x in range(len(pre)):
                val = pre[x]
                right = left
                cur = nums[left]
                for y in range(left, ceil):
                    cur &= nums[y]
                    if cur >= val:
                        right = y
                    else:
                        break
                assert right == st1_and.bisect_right(left, val, (1 << 32) - 1)[0]
        return

    def test_sparse_table_2d_max_min(self):

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
