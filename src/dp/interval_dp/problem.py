"""
Algorithm：interval_dp
Description：prefix_sum|interval_dp|preprocess_dp|memory_search

====================================LeetCode====================================
375（https://leetcode.com/problems/guess-number-higher-or-lower-ii/）interval_dp
1039（https://leetcode.com/problems/minimum-score-triangulation-of-polygon/）circular_array|interval_dp
2472（https://leetcode.com/problems/maximum-number-of-non-overlapping-palindrome-substrings/）palindrome_substring|linear_dp|manacher
2430（https://leetcode.com/problems/maximum-deletions-on-a-string/）lcp|liner_dp
1547（https://leetcode.com/problems/minimum-cost-to-cut-a-stick/）interval_dp|implemention
1278（https://leetcode.com/problems/palindrome-partitioning-iii/）preprocess_dp|interval_dp
1690（https://leetcode.com/problems/stone-game-vii/description/）interval_dp
1312（https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/）interval_dp|longest_palindrome_subsequence

=====================================LuoGu======================================
1521（https://www.luogu.com.cn/problem/P1521）merge_sort|multiplication_method|tree_array
1775（https://www.luogu.com.cn/problem/P1775）classical|interval_dp|prefix_sum
2426（https://www.luogu.com.cn/problem/P2426）classical|interval_dp
2690（https://www.luogu.com.cn/problem/P2690）interval_dp|implemention|memory_search
1435（https://www.luogu.com.cn/problem/P1435）classical|interval_dp|longest_non_sub_consequence_palindrome
1388（https://www.luogu.com.cn/problem/P1388）back_track|brute_force|interval_dp
1103（https://www.luogu.com.cn/problem/P1103）matrix_dp
2858（https://www.luogu.com.cn/problem/P2858）classical|interval_dp
1880（https://www.luogu.com.cn/problem/P1880）circular|interval_dp
3205（https://www.luogu.com.cn/problem/P3205）interval_dp|rolling_update
1040（https://www.luogu.com.cn/problem/P1040）interval_dp|specific_plan
1430（https://www.luogu.com.cn/problem/P1430）interval_dp|prefix_sum
2308（https://www.luogu.com.cn/problem/P2308）interval_dp|recursion
2734（https://www.luogu.com.cn/problem/P2734）prefix_sum|interval_dp
3004（https://www.luogu.com.cn/problem/P3004）interval_dp
3205（https://www.luogu.com.cn/problem/P3205）interval_dp|rolling_update
4170（https://www.luogu.com.cn/problem/P4170）interval_dp|math

===================================CodeForces===================================
1509C（https://codeforces.com/problemset/problem/1509/C）interval_dp
607B（https://codeforces.com/problemset/problem/607/B）interval_dp

=====================================AcWing=====================================
3996（https://www.acwing.com/problem/content/3999/）interval_dp|longest_palindrome_subsequence

"""
from functools import lru_cache
from itertools import accumulate
from math import inf
from typing import List

from src.utils.fast_io import FastIO

MOD = 10 ** 9 + 7


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_307b(ac=FastIO()):
        #
        n = ac.read_int()
        nums = ac.read_list_ints()

        # 初始化
        dp = [[inf] * n for _ in range(n + 1)]
        for i in range(n):
            for j in range(i):
                dp[i][j] = 0
        dp[n] = [0] * n

        # 状态转移
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            if i + 1 < n:
                dp[i][i + 1] = 2 if nums[i] != nums[i + 1] else 1
            for j in range(i + 2, n):

                dp[i][j] = ac.min(dp[i + 1][j], dp[i][j - 1]) + 1
                if nums[i] == nums[i + 1]:
                    dp[i][j] = ac.min(dp[i][j], 1 + dp[i + 2][j])

                for k in range(i + 2, j + 1):
                    dp[i][j] = ac.min(dp[i][j], dp[i][k] + dp[k + 1][j])
                    if nums[k] == nums[i]:
                        dp[i][j] = ac.min(dp[i][j], dp[i + 1][k - 1] + dp[k + 1][j])

        ac.st(dp[0][n - 1])
        return

    @staticmethod
    def cf_1509c(n, nums):
        # 数组interval_dp|转移求解
        dp = [[inf] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 0
            for j in range(i + 1, n):
                dp[i][j] = nums[j] - nums[i] + min(dp[i + 1][j], dp[i][j - 1])
        return dp[0][n - 1]

    @staticmethod
    def lc_1312(s: str) -> int:
        """
        url: https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/
        tag: interval_dp|longest_palindrome_subsequence
        """
        # interval_dp|，最长回文子序列
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            if i + 1 < n:
                dp[i][i + 1] = 2 if s[i] == s[i + 1] else 1
            for j in range(i + 2, n):
                a, b = dp[i + 1][j], dp[i][j - 1]
                a = a if a > b else b
                b = dp[i + 1][j - 1] + 2 * int(s[i] == s[j])

                dp[i][j] = a if a > b else b
        return n - dp[0][n - 1]

    @staticmethod
    def lc_1547(n: int, cuts: List[int]) -> int:
        """
        url: https://leetcode.com/problems/minimum-cost-to-cut-a-stick/
        tag: interval_dp|implemention
        """
        # interval_dp|implemention
        cuts.sort()
        cuts.insert(0, 0)
        cuts.append(n)
        m = len(cuts)
        dp = [[0] * (m + 1) for _ in range(m + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(i + 2, m):
                dp[i][j] = cuts[j] - cuts[i] + min(dp[i][k] + dp[k][j] for k in range(i + 1, j))
        return dp[0][m - 1]

    @staticmethod
    def lc_1690(stones: List[int]) -> int:
        """
        url: https://leetcode.com/problems/stone-game-vii/description/
        tag: interval_dp
        """
        # interval_dp|
        n = len(stones)
        pre = list(accumulate(stones, initial=0))

        @lru_cache(None)
        def dfs(i, j):
            if i == j:
                return 0
            return max(-dfs(i + 1, j) + pre[j + 1] - pre[i + 1], -dfs(i, j - 1) + pre[j] - pre[i])

        ans = dfs(0, n - 1)
        dfs.cache_clear()
        return ans

    @staticmethod
    def lc_2472(s: str, k: int) -> int:
        """
        url: https://leetcode.com/problems/maximum-number-of-non-overlapping-palindrome-substrings/
        tag: palindrome_substring|linear_dp|manacher
        """
        # preprocess线性palindrome_substring DP 优化外|结果linear_dp 也可以manacher回文串获取回文信息
        n = len(s)
        res = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            res[i][i] = 1
            if i + 1 < n:
                res[i][i + 1] = 1 if s[i] == s[i + 1] else 0
            for j in range(i + 2, n):
                if s[i] == s[j] and res[i + 1][j - 1]:
                    res[i][j] = 1

        dp = [0] * (n + 1)
        for i in range(n):
            dp[i + 1] = dp[i]
            for j in range(0, i - k + 2):
                if i - j + 1 >= k and res[j][i]:
                    dp[i + 1] = max(dp[i + 1], dp[j] + 1)
        return dp[-1]

    @staticmethod
    def lg_p3205(ac=FastIO()):
        # interval_dp|滚动数组更新
        n = ac.read_int()
        nums = ac.read_list_ints()
        mod = 19650827
        dp = [[[0, 0] for _ in range(n)] for _ in range(2)]
        pre = 0
        for i in range(n - 1, -1, -1):
            cur = 1 - pre
            dp[cur][i][0] = 1
            for j in range(i + 1, n):
                x = 0
                # 后 j
                if nums[j - 1] < nums[j]:
                    x += dp[cur][j - 1][1]
                if nums[i] < nums[j]:
                    x += dp[cur][j - 1][0]
                dp[cur][j][1] = x % mod
                x = 0
                # 后 i
                if nums[i + 1] > nums[i]:
                    x += dp[pre][j][0]
                if nums[j] > nums[i]:
                    x += dp[pre][j][1]
                dp[cur][j][0] = x % mod
            pre = cur
        ac.st(sum(dp[pre][n - 1]) % mod)
        return

    @staticmethod
    def lg_p1040(ac=FastIO()):
        # interval_dp|最小代价，并迭代还原前序遍历路径
        n = ac.read_int()
        nums = ac.read_list_ints()

        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = nums[i]
            if i + 1 < n:
                dp[i][i + 1] = nums[i] + nums[i + 1]
            for j in range(i + 2, n):
                dp[i][j] = max(dp[i][k - 1] * dp[k + 1][j] + dp[k][k] for k in range(i + 1, j))

        # stackimplementionspecific_plan还原
        ans = []
        stack = [[0, n - 1]]
        while stack:
            i, j = stack.pop()
            if i == j:
                ans.append(i + 1)
                continue
            if i == j - 1:
                ans.append(i + 1)
                ans.append(j + 1)
                continue
            for k in range(i + 1, j):
                if dp[i][j] == dp[i][k - 1] * dp[k + 1][j] + dp[k][k]:
                    ans.append(k + 1)
                    stack.append([k + 1, j])
                    stack.append([i, k - 1])
                    break
        ac.st(dp[0][n - 1])
        ac.lst(ans)
        return

    @staticmethod
    def lg_p1430(ac=FastIO()):
        # interval_dp||前缀数组优化
        for _ in range(ac.read_int()):
            nums = ac.read_list_ints()
            n = nums.pop(0)
            pre = [0] * (n + 1)
            for i in range(n):
                pre[i + 1] = pre[i] + nums[i]

            dp = [0] * n
            post = [0] * n
            for i in range(n - 1, -1, -1):
                dp[i] = nums[i]
                post[i] = ac.min(nums[i], post[i])
                floor = ac.min(0, nums[i])
                for j in range(i + 1, n):
                    s = pre[j + 1] - pre[i]
                    dp[j] = s
                    dp[j] = ac.max(dp[j], s - post[j])
                    dp[j] = ac.max(dp[j], s - floor)
                    floor = ac.min(floor, dp[j])
                    post[j] = ac.min(post[j], dp[j])
            ac.st(dp[n - 1])
        return

    @staticmethod
    def lg_p2308(ac=FastIO()):
        # interval_dp|，并recursion方式反解括号添|方式以及每一步的和
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = ac.accumulate(nums)
        # 记录转移的中间节点与|和最小值
        dp = [[inf] * n for _ in range(n)]
        mid = [[-1] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 0
            for j in range(i + 1, n):
                ind = i
                for k in range(i, j):
                    cur = dp[i][k] + dp[k + 1][j] + pre[j + 1] - pre[i]
                    if cur < dp[i][j]:
                        dp[i][j] = cur
                        ind = k
                mid[i][j] = ind

        # 反推路径
        ans = []
        nums = [str(x) for x in nums]
        stack = [[0, n - 1]]
        while stack:
            i, j = stack.pop()
            if i >= 0:
                stack.append([~i, j])
                if i >= j - 1:
                    continue
                k = mid[i][j]
                # 先左后右
                stack.append([k + 1, j])
                stack.append([i, k])
            else:
                i = ~i
                if i < j:
                    nums[i] = "(" + nums[i]
                    nums[j] = nums[j] + ")"
                    ans.append(pre[j + 1] - pre[i])
        ac.st("+".join(nums))
        ac.st(sum(ans))
        ac.lst(ans)
        return

    @staticmethod
    def lg_p2734(ac=FastIO()):

        # prefix_sum|interval_dp

        n = ac.read_int()
        nums = []
        while len(nums) < n:
            nums.extend(ac.read_list_ints())
        pre = ac.accumulate(nums)

        # @lru_cache(None)
        # def dfs(i, j):
        #     if i == j:
        #         return nums[j]
        #     res = nums[i]+pre[j+1]-pre[i+1]-dfs(i+1, j)
        #     res = ac.max(res, nums[j]+pre[j]-pre[i]-dfs(i, j-1))
        #     return res
        # a = dfs(0, n-1)
        # ac.lst([a, pre[-1]-a])

        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = nums[i]
            for j in range(i + 1, n):
                dp[i][j] = ac.max(nums[i] + pre[j + 1] - pre[i + 1] - dp[i + 1][j],
                                  nums[j] + pre[j] - pre[i] - dp[i][j - 1])
        a = dp[0][n - 1]
        ac.lst([a, pre[-1] - a])
        return

    @staticmethod
    def lg_p3004(ac=FastIO()):
        # 简单interval_dp
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        dp = [[0] * n for _ in range(2)]
        pre = ac.accumulate(nums)
        x = 0
        for i in range(n - 1, -1, -1):
            y = 1 - x
            dp[y][i] = nums[i]
            for j in range(i + 1, n):
                dp[y][j] = ac.max(pre[j + 1] - pre[i + 1] - dp[x][j] + nums[i],
                                  pre[j] - pre[i] - dp[y][j - 1] + nums[j])
            x = y
        ac.st(dp[x][n - 1])
        return

    @staticmethod
    def lg_p4170(ac=FastIO()):
        # interval_dp 注意转移方程
        s = ac.read_str()
        n = len(s)
        dp = [[inf] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = ac.min(dp[i + 1][j], dp[i][j - 1])
                else:
                    for k in range(i, j):
                        dp[i][j] = ac.min(dp[i][j], dp[i][k] + dp[k + 1][j])
        ac.st(dp[0][n - 1])
        return

    @staticmethod
    def ac_3996(ac=FastIO()):
        # interval_dp 最长回文子序列变形
        ac.read_int()
        nums = ac.read_list_ints()
        pre = []
        for num in nums:
            if pre and pre[-1] == num:
                continue
            pre.append(num)
        nums = pre[:]
        n = len(nums)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if nums[i] == nums[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 1
                else:
                    dp[i][j] = ac.min(dp[i + 1][j], dp[i][j - 1]) + 1
        ac.st(dp[0][n - 1])
        return

    @staticmethod
    def lc_1278(s: str, k: int) -> int:
        """
        url: https://leetcode.com/problems/palindrome-partitioning-iii/
        tag: preprocess_dp|interval_dp
        """
        # preprocess双重interval_dp|
        n = len(s)

        cost = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            if i + 1 < n:
                j = i + 1
                cost[i][j] = 1 if s[i] != s[j] else 0
            for j in range(i + 2, n):
                cost[i][j] = cost[i + 1][j - 1] + int(s[i] != s[j])

        dp = [[n] * k for _ in range(n + 1)]
        for i in range(n):
            dp[i + 1][0] = cost[0][i]
            for j in range(1, k):
                dp[i + 1][j] = min(dp[x][j - 1] + cost[x][i] for x in range(i + 1))
        return dp[n][k - 1]