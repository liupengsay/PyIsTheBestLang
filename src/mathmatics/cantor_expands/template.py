from src.data_structure.sorted_list.template import SortedList
from src.data_structure.tree_array.template import PointAddRangeSum


class CantorExpands:
    def __init__(self, n, mod=0):
        self.mod = mod
        self.perm = [1] * (n + 1)
        for i in range(2, n):
            if mod:
                self.perm[i] = i * self.perm[i - 1] % mod
            else:
                self.perm[i] = i * self.perm[i - 1]
        return

    def array_to_rank(self, nums):
        """"permutation rank of nums"""
        n = len(nums)
        out = 1
        lst = SortedList(nums)
        for i in range(n):
            fact = self.perm[n - i - 1]
            res = lst.bisect_left(nums[i])
            lst.discard(nums[i])
            out += res * fact
            if self.mod:
                out %= self.mod
        return out

    def array_to_rank_with_tree(self, nums):
        """"permutation rank of nums"""
        n = len(nums)
        out = 1
        tree = PointAddRangeSum(n)
        tree.build([1] * n)
        for i in range(n):
            fact = self.perm[n - i - 1]
            res = tree.range_sum(0, nums[i] - 2) if nums[i] >= 2 else 0
            tree.point_add(nums[i] - 1, -1)
            out += res * fact
            if self.mod:
                out %= self.mod
        return out


    def rank_to_array(self, n, k):
        """"nums with permutation rank k"""
        nums = list(range(1, n + 1))
        ans = []
        while k and nums:
            single = self.perm[len(nums) - 1]
            i = (k - 1) // single
            ans.append(nums.pop(i))
            k -= i * single
        return ans
