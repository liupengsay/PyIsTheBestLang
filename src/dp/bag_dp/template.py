import math
from collections import defaultdict




class BagDP:
    def __init__(self):
        return

    @staticmethod
    def bin_split_1(num):
        # binary optimization refers to continuous operations such as 1.2.4.x
        # instead of binary 10101 corresponding to 1
        if not num:
            return []
        lst = []
        x = 1
        while x <= num:
            lst.append(x)
            num -= x
            x *= 2
        if num:
            lst.append(num)
        return lst

    @staticmethod
    def bin_split_2(num):
        # split from large to small to ensure that there are no identical positive numbers other than 1
        if not num:
            return []
        lst = []
        while num:
            lst.append((num + 1) // 2)
            num //= 2
        lst.reverse()
        return lst

    @staticmethod
    def one_dimension_limited(n, nums):
        # 01 backpack
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for i in range(n, num - 1, -1):
                dp[i] += dp[i - num]
        return dp[n]

    @staticmethod
    def one_dimension_unlimited(n, nums):
        # complete backpack
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for i in range(num, n + 1):
                dp[i] += dp[i - num]
        return dp[n]

    @staticmethod
    def two_dimension_limited(m, n, nums):
        # 2D 01 backpack
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        for a, b in nums:
            for i in range(m, a - 1, -1):
                for j in range(n, b - 1, -1):
                    dp[i][j] += dp[i - a][j - b]
        return dp[m][n]

    @staticmethod
    def two_dimension_unlimited(m, n, nums):
        # 2D complete backpack
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        for a, b in nums:
            for i in range(a, m + 1):
                for j in range(b, n + 1):
                    dp[i][j] += dp[i - a][j - b]
        return dp[m][n]

    def continuous_bag_with_bin_split(self, n, nums):
        # Continuous 01 Backpack Using Binary Optimization
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for x in self.bin_split_1(num):
                for i in range(n, x - 1, -1):
                    dp[i] += dp[i - x]
        return dp[n]

    @staticmethod
    def group_bag_limited(n, d, nums):
        # group backpack
        pre = [math.inf] * (n + 1)
        pre[0] = 0
        for r, z in nums:
            cur = pre[:]  # The key is that we need group backpacks here
            for x in range(1, z + 1):
                cost = d + x * r
                for i in range(n, x - 1, -1):
                    if pre[i - x] + cost < cur[i]:
                        cur[i] = pre[i - x] + cost
            pre = cur[:]
        if pre[n] < inf:
            return pre[n]
        return -1

    @staticmethod
    def group_bag_unlimited(nums):
        # Calculate the number of solutions for decomposing n into the sum of squares of four numbers
        n = max(nums)
        dp = [[0] * 5 for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, int(math.sqrt(n)) + 1):
            x = i * i
            for j in range(x, n + 1):
                for k in range(1, 5):
                    if dp[j - x][k - 1]:
                        dp[j][k] += dp[j - x][k - 1]
        return [sum(dp[num]) for num in nums]

    @staticmethod
    def one_dimension_limited_use_dct(nums):
        # One dimensional finite backpack
        # using a dictionary for transfer records with negative numbers
        pre = defaultdict(lambda: -inf)
        # can also use [0]*2*s where s is the sum(abs(x) for x in nums)
        pre[0] = 0
        for s, f in nums:
            cur = pre.copy()
            for p in pre:
                cur[p + s] = max(cur[p + s], pre[p] + f)
            pre = cur
        ans = 0
        for p in pre:
            if p >= 0 and pre[p] >= 0:
                ans = ans if ans > p + pre[p] else p + pre[p]
        return ans
