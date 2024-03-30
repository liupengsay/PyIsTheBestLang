import random
import unittest
from functools import reduce
from operator import xor

from src.mathmatics.linear_basis.template import LinearBasis


class TestGeneral(unittest.TestCase):

    def test_linear_basis(self):
        for x in range(1000):
            m = 10
            lst = [random.randint(0, 1000000) for _ in range(m)]
            if x == 0:
                lst = [0]
                m = 1
            nums = [0]
            zero = 0
            for i in range(1, 1 << m):
                nums.append(reduce(xor, [lst[j] for j in range(m) if i & (1 << j)]))
                if not nums[-1]:
                    zero = 1
            nums = sorted(set(nums))
            lb = LinearBasis(20)
            for num in lst:
                lb.add(num)
            assert len(nums) == lb.tot
            assert lb.zero == zero
            n = len(nums)
            for i in range(n):
                assert lb.query_kth_xor(i) == nums[i]
                assert lb.query_xor_kth(nums[i]) == i

            x = random.randint(0, 1000)
            lst.append(x)
            m += 1
            zero = 0
            nums = [0]

            for i in range(1, 1 << m):
                nums.append(reduce(xor, [lst[j] for j in range(m) if i & (1 << j)]))
                if not nums[-1]:
                    zero = 1
            nums = sorted(set(nums))
            lb.add(x)
            assert len(nums) == lb.tot
            assert lb.zero == zero
            n = len(nums)
            for i in range(n):
                assert lb.query_kth_xor(i) == nums[i]
                assert lb.query_xor_kth(nums[i]) == i
            assert lb.query_max() == nums[-1]
            assert lb.query_min() == nums[0]
        return


if __name__ == '__main__':
    unittest.main()
