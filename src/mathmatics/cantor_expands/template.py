from src.data_structure.sorted_list.template import SortedList


class CantorExpands:
    def __init__(self, n, mod=10 ** 9 + 7):
        self.mod = mod
        self.dp = [1] * (n + 1)
        for i in range(2, n):
            self.dp[i] = i * self.dp[i - 1] % mod
        return

    def array_to_rank(self, nums):
        """"permutation rank of nums"""
        n = len(nums)
        out = 1
        lst = SortedList(nums)
        for i in range(n):
            fact = self.dp[n - i - 1]
            res = lst.bisect_left(nums[i])
            lst.discard(nums[i])
            out += res * fact
            out %= self.mod
        return out

    def rank_to_array(self, n, k):
        """"nums with permutation rank k"""
        nums = list(range(1, n + 1))
        ans = []
        while k and nums:
            single = self.dp[len(nums) - 1]
            i = (k - 1) // single
            ans.append(nums.pop(i))
            k -= i * single
        return ans
