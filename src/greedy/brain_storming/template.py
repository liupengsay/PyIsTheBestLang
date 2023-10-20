import math


class BrainStorming:
    def __init__(self):
        return

    @staticmethod
    def minimal_coin_need(n, m, nums):

        nums += [m + 1]
        nums.sort()
        # 有 n 个可选取且无限的硬币，为了形成 1-m 所有组合需要的最少硬币个数
        if nums[0] != 1:
            return -1
        ans = sum_ = 0
        for i in range(n):
            nex = nums[i + 1] - 1
            nex = nex if nex < m else m
            x = math.ceil((nex - sum_) / nums[i])
            x = x if x >= 0 else 0
            ans += x
            sum_ += x * nums[i]
            if sum_ >= m:
                break
        return ans
