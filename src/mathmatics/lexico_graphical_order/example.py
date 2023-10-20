import random
import unittest
from itertools import permutations, combinations

from mathmatics.lexico_graphical_order.template import LexicoGraphicalOrder


class TestGeneral(unittest.TestCase):

    def test_lexico_graphical_order(self):
        lgo = LexicoGraphicalOrder()

        n = 10**5
        nums = sorted([str(x) for x in range(1, n + 1)])
        for _ in range(100):
            i = random.randint(0, n - 1)
            num = nums[i]
            assert lgo.get_kth_num(n, i + 1) == int(num)
            assert lgo.get_num_kth(n, int(num)) == i + 1

        n = 10
        nums = []
        for i in range(1 << n):
            nums.append([j + 1 for j in range(n) if i & (1 << j)])
        nums.sort()
        nums[0] = [0]
        for _ in range(100):
            i = random.randint(0, n - 1)
            lst = nums[i]
            assert lgo.get_kth_subset(n, i + 1) == lst
            assert lgo.get_subset_kth(n, lst) == i + 1

        n = 10
        m = 4
        nums = []
        for item in combinations(list(range(1, n+1)), m):
            nums.append(list(item))
        for _ in range(100):
            i = random.randint(0, len(nums) - 1)
            lst = nums[i]
            assert lgo.get_kth_subset_comb(n, m, i+1) == lst
            assert lgo.get_subset_comb_kth(n, m, lst) == i + 1

        n = 8
        nums = []
        for item in permutations(list(range(1, n+1)), n):
            nums.append(list(item))
        for i, lst in enumerate(nums):
            lst = nums[i]
            assert lgo.get_kth_subset_perm(n, i + 1) == lst
            assert lgo.get_subset_perm_kth(n, lst) == i + 1
        return


if __name__ == '__main__':
    unittest.main()
