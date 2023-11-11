class CantorExpands:
    def __init__(self, n, mod=10 ** 9 + 7):
        self.mod = mod
        self.dp = [1] * (n + 1)
        for i in range(2, n):
            self.dp[i] = i * self.dp[i - 1] % mod
        return

    def array_to_rank(self, nums):
        """"permutation rank of nums"""
        lens = len(nums)
        out = 1
        for i in range(lens):
            res = 0
            fact = self.dp[lens - i - 1]
            for j in range(i + 1, lens):
                if nums[j] < nums[i]:
                    res += 1
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
