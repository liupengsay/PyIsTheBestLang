import unittest

from functools import reduce
from operator import xor

from src.mathmatics.linear_basis.template import LinearBasis


class TestGeneral(unittest.TestCase):
    def test_euler_phi(self):
        lst = [0, 1, 2, 4, 8, 16]
        m = len(lst)

        # 计算所有的异或和
        nums = []
        for i in range(1, 1 << m):
            nums.append(reduce(xor, [lst[j] for j in range(m) if i & (1 << j)]))
        nums = sorted(set(nums))

        # 查询最大最小以及是否存在对应异或值
        lb = LinearBasis(lst)
        assert lb.query_max() == 31
        assert lb.query_min() == 0
        assert lb.query_xor(20)

        # 查询第 k 小以及异或和是第几小
        n = len(nums)
        for i in range(n):
            assert lb.query_k_rank(i + 1) == nums[i]
            assert lb.query_k_smallest(nums[i]) == i + 1

        # 超出范围
        assert lb.query_k_rank(len(nums) + 1) == -1
        assert lb.query_k_smallest(nums[-1] + 1) == -1
        return


if __name__ == '__main__':
    unittest.main()
