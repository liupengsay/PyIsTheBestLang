class LinearDP:
    def __init__(self):
        return

    @staticmethod
    def liner_dp_template(nums):
        # example of lis（longest increasing sequence）
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = 1
            for j in range(i):
                if nums[i] > nums[j] and dp[j] + 1 > dp[i + 1]:
                    dp[i + 1] = dp[j] + 1
        return max(dp)
