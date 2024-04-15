"""
Algorithm：circular_array|liner_dp|interval_dp|brute_force|circle_to_linear
Description：operations on circular_array, always copy the array to double or use index to implement liner_dp


====================================LeetCode====================================
213（https://leetcode.cn/problems/house-robber-ii/）circular_array|linear_dp|classical
918（https://leetcode.cn/problems/maximum-sum-circular-subarray/）circular_array|brute_force|sub_array
1388（https://leetcode.cn/problems/pizza-with-3n-slices/）brute_force|circular_array|linear_dp
1888（https://leetcode.cn/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/）circular_dp|brute_force
2560（https://leetcode.cn/problems/house-robber-iv/）binary_search|circular_array|linear_dp|classical

=====================================LuoGu======================================
P1880（https://www.luogu.com.cn/problem/P1880）brute_force|circular_array|linear_dp
P1121（https://www.luogu.com.cn/problem/P1121）circular_array|brute_force|sub_array
P1043（https://www.luogu.com.cn/problem/P1043）brute_force|circular_array|linear_dp
P1133（https://www.luogu.com.cn/problem/P1133）brute_force|circular_array|linear_dp

=====================================AtCoder======================================
ABC251E（https://atcoder.jp/contests/abc251/tasks/abc251_e）circular_array|linear_dp|classical


"""
from collections import defaultdict
from math import inf
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def abc_251e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc251/tasks/abc251_e
        tag: circular_array|linear_dp|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        dp = [inf] * n
        dp[0] = nums[-1]
        dp[1] = dp[0] + nums[0]
        for i in range(2, n):
            dp[i] = min(dp[i - 1], dp[i - 2]) + nums[i - 1]
        ans1 = min(dp[n - 2], dp[n - 3] + nums[n - 2])

        dp = [inf] * n
        dp[0] = nums[0]
        dp[1] = nums[0]
        for i in range(2, n):
            dp[i] = min(dp[i - 1], dp[i - 2]) + nums[i - 1]
        ans2 = min(dp[n - 1], dp[n - 2] + nums[n - 1])

        ans = min(ans1, ans2)
        ac.st(ans)
        return

    @staticmethod
    def lc_213(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/house-robber-ii
        tag: circular_array|linear_dp|classical
        """

        def check(lst):
            n = len(lst)
            dp = [0] * (n + 1)
            for i in range(n):
                dp[i + 1] = dp[i] if dp[i] > lst[i] else lst[i]
                if i and dp[i - 1] + lst[i] > dp[i + 1]:
                    dp[i + 1] = dp[i - 1] + lst[i]
            return dp[-1]

        if len(nums) == 1:
            return nums[0]
        return max(check(nums[1:]), check(nums[:-1]))

    @staticmethod
    def lc_1388(slices: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/pizza-with-3n-slices/
        tag: brute_force|circular_array|linear_dp
        """

        def check(lst):
            n = len(lst)
            m = n // 3 + 1
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            for i in range(n):
                for j in range(1, m + 1):
                    dp[i + 1][j] = max(dp[i][j], dp[i - 1][j - 1] + lst[i] if i else lst[i])
            return dp[-1][-1]

        ans = max(check(slices[1:]), check(slices[:-1]))

        return ans

    @staticmethod
    def lg_p1880(ac=FastIO()):

        """
        url: https://www.luogu.com.cn/problem/P1880
        tag: brute_force|circular_array|linear_dp
        """

        def check(fun):
            dp = [[0] * n for _ in range(n)]
            for i in range(n - 1, -1, -1):
                dp[i][i] = 0
                if i + 1 < n:
                    dp[i][i + 1] = nums[i] + nums[i + 1]
                for j in range(i + 2, n):
                    dp[i][j] = 0 if fun == max else inf
                    for k in range(i, j):
                        cur = dp[i][k] + dp[k + 1][j] + pre[j + 1] - pre[i]
                        dp[i][j] = fun(dp[i][j], cur)
            return fun([dp[i][i + n // 2 - 1] for i in range(n // 2)])

        n = ac.read_int() * 2
        nums = ac.read_list_ints()
        nums.extend(nums)
        pre = [0] * (n + 1)
        for x in range(n):
            pre[x + 1] = pre[x] + nums[x]
        ac.st(check(min))
        ac.st(check(max))
        return

    @staticmethod
    def lg_p1121(ac=FastIO()):

        """
        url: https://www.luogu.com.cn/problem/P1121
        tag: circular_array|brute_force|sub_array
        """

        n = ac.read_int()
        nums = ac.read_list_ints()
        s = sum(nums)

        pre = [-inf] * (n + 1)
        x = 0
        for i in range(n):
            x = x if x > 0 else 0
            x += nums[i]
            pre[i + 1] = ac.max(pre[i], x)

        post = [-inf] * (n + 1)
        x = 0
        for i in range(n - 1, -1, -1):
            x = x if x > 0 else 0
            x += nums[i]
            post[i] = ac.max(post[i + 1], x)
        ans = max(pre[i] + post[i + 1] for i in range(1, n))
        cnt = sum(num >= 0 for num in nums)
        if cnt <= 1:
            ac.st(ans)
            return

        pre = [0] * (n + 1)
        x = 0
        for i in range(n):
            x = x if x < 0 else 0
            x += nums[i]
            pre[i + 1] = ac.min(pre[i], x)

        post = [0] * (n + 1)
        x = 0
        for i in range(n - 1, -1, -1):
            x = x if x < 0 else 0
            x += nums[i]
            post[i] = ac.min(post[i + 1], x)

        ans = ac.max(ans, s - min(pre[i] + post[i + 1] for i in range(1, n)))
        ac.st(ans)
        return

    @staticmethod
    def lc_918(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-sum-circular-subarray/
        tag: circular_array|brute_force|sub_array
        """

        ans = -inf
        pre = 0
        for num in nums:
            pre = pre if pre > 0 else 0
            pre += num
            if pre > ans:
                ans = pre

        low = inf
        pre = 0
        for num in nums:
            pre = pre if pre < 0 else 0
            pre += num
            if pre < low:
                low = pre

        if all(num < 0 for num in nums):
            return ans
        low = sum(nums) - low
        if low > ans:
            ans = low
        return ans

    @staticmethod
    def lc_2560(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/house-robber-iv/
        tag: binary_search|circular_array|linear_dp|classical
        """
        n = len(nums)

        def check(x):
            dp = [0] * (n + 1)
            for i in range(n):
                dp[i + 1] = dp[i]
                if nums[i] <= x and dp[i - 1] + 1 > dp[i + 1]:
                    dp[i + 1] = dp[i - 1] + 1
            return dp[-1] >= k

        return BinarySearch().find_int_left(min(nums), max(nums), check)

    @staticmethod
    def lg_p1043(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1043
        tag: brute_force|circular_array|linear_dp
        """
        n, m = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]
        floor = inf
        ceil = -inf
        for _ in range(n):
            nums.append(nums.pop(0))
            pre = [0] * (n + 1)
            for i in range(n):
                pre[i + 1] = pre[i] + nums[i]
            dp = [[[inf, -inf] for _ in range(m + 1)] for _ in range(n + 1)]
            dp[0][0] = [1, 1]
            for i in range(n):
                for k in range(i + 1):
                    for j in range(1, m + 1):
                        cur = (pre[i + 1] - pre[k]) % 10
                        if dp[k][j - 1][0] != inf:
                            dp[i + 1][j][0] = min(dp[i + 1][j][0], dp[k][j - 1][0] * cur)
                            dp[i + 1][j][1] = max(dp[i + 1][j][1], dp[k][j - 1][0] * cur)
                        if dp[k][j - 1][1] != -inf:
                            dp[i + 1][j][0] = min(dp[i + 1][j][0], dp[k][j - 1][1] * cur)
                            dp[i + 1][j][1] = max(dp[i + 1][j][1], dp[k][j - 1][1] * cur)
            floor = min(floor, dp[-1][-1][0])
            ceil = max(ceil, dp[-1][-1][1])
        ac.st(floor)
        ac.st(ceil)
        return

    @staticmethod
    def lg_p1133(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1133
        tag: brute_force|circular_array|linear_dp
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        ans = 0
        state = []
        for a in range(3):
            for b in range(3):
                if a != b:
                    state.append([a, b])
        nex = defaultdict(list)
        for a, b in state:
            for c in range(3):
                if b < a and b < c:
                    nex[(a, b)].append(c)
                if b > a and b > c:
                    nex[(a, b)].append(c)

        for a, b in state:
            pre = dict()
            pre[(a, b)] = nums[-1][a] + nums[0][b]
            for i in range(1, n - 1):
                cur = dict()
                for x1, x2 in pre:
                    for y in nex[(x1, x2)]:
                        cur[(x2, y)] = ac.max(cur.get((x2, y), 0), pre[(x1, x2)] + nums[i][y])
                pre = cur.copy()
            for x1, x2 in pre:
                if (a < x2 and a < b) or (a > x2 and a > b):
                    ans = ac.max(ans, pre[(x1, x2)])
        ac.st(ans)
        return
