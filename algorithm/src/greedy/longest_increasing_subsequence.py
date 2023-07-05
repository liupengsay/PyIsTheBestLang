import bisect
import unittest
from bisect import bisect_left
from collections import deque, defaultdict
from typing import List

from algorithm.src.fast_io import FastIO

"""

算法：最长上升（或不降）子序列 Longest Increasing Subsequence（LIS）Longest Decreasing Subsequence（LDS）统称 Longest Monotonic Subsequence（LMS） 
最长单调递增子序列（严格上升）：<
最长单调不减子序列（不降）：<=
最长单调递减子序列（严格下降）：>
最长单调不增子序列（不升）：>=
对于数组来说，正数反可以将后两个问题3和4转换为前两个问题1和2进行解决，可以算全局的最长单调子序列，也可以计算前后缀的最长单调子序列
dilworth定理：分成不下降子序列最小组数等于最大上升子序列的长度，分成不上升子序列最小组数等于最大下降子序列的长度。
参考题目：
===================================力扣===================================
354. 俄罗斯套娃信封问题（https://leetcode.cn/problems/russian-doll-envelopes/）经典二维偏序最长递增子序列问题
673. 最长递增子序列的个数（https://leetcode.cn/problems/number-of-longest-increasing-subsequence/）经典O(n^2)与O(nlogn)的LIS计数问题
1092. 最短公共超序列（https://leetcode.cn/problems/shortest-common-supersequence/）经典利用LIS求LCS的最短公共超序列
1671. 得到山形数组的最少删除次数（https://leetcode.cn/problems/minimum-number-of-removals-to-make-mountain-array/）经典山脉数组LIS变形问题
2111. 使数组 K 递增的最少操作次数（https://leetcode.cn/problems/minimum-operations-to-make-the-array-k-increasing/）分成 K 组计算每组的最长递增子序列
面试题 17.08. 马戏团人塔（https://leetcode.cn/problems/circus-tower-lcci/）按照两个维度贪心排序后，计算最长递增子序列
最长递增子序列（https://www.nowcoder.com/questionTerminal/30fb9b3cab9742ecae9acda1c75bf927?orderByHotValue=1&questionTypes=000100&difficulty=11111&mutiTagIds=593&page=10&onlyReference=false）最长且字典序最小的递增子序列

===================================洛谷===================================
P1020 导弹拦截（https://www.luogu.com.cn/problem/P1020）使用贪心加二分计算最长单调不减和单调不增子序列的长度
P1439 【模板】最长公共子序列（https://www.luogu.com.cn/problem/P1439）使用贪心加二分计算最长单调递增子序列的长度
P1091 合唱队形（https://www.luogu.com.cn/problem/P1091）可以往前以及往后计算最长单调子序列
P1233 木棍加工（https://www.luogu.com.cn/problem/P1233）按照一个维度排序后计算另一个维度的，最长严格递增子序列的长度
P2782 友好城市（https://www.luogu.com.cn/problem/P2782）按照一个维度排序后计算另一个维度的，最长严格递增子序列的长度（也可以考虑使用线段树求区间最大值）
P3902 递增（https://www.luogu.com.cn/problem/P3902）最长严格上升子序列
P6403 [COCI2014-2015#2] STUDENTSKO（https://www.luogu.com.cn/problem/P6403）问题转化为最长不降子序列
P5939 [POI1998]折线（https://www.luogu.com.cn/problem/P5939）旋转后转换为 LIS 问题
P5978 [CEOI2018] Global warming（https://www.luogu.com.cn/problem/P5978）经典 LIS 变形问题，贪心枚举前半部分
P7957 [COCI2014-2015#6] KRATKI（https://www.luogu.com.cn/problem/P7957）经典 LMS 逆问题构造

"""


class LongestIncreasingSubsequence:
    def __init__(self):
        return

    @staticmethod
    def definitely_increase(nums):
        # 最长单调递增子序列（严格上升）
        dp = []
        for num in nums:
            i = bisect.bisect_left(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    @staticmethod
    def definitely_not_reduce(nums):
        # 最长单调不减子序列（不降）
        dp = []
        for num in nums:
            i = bisect.bisect_right(dp, num)
            if 0 <= i < len(dp):
                dp[i] = num
            else:
                dp.append(num)
        return len(dp)

    def definitely_reduce(self, nums):
        # 最长单调递减子序列（严格下降）
        nums = [-num for num in nums]
        return self.definitely_increase(nums)

    def definitely_not_increase(self, nums):
        # 最长单调不增子序列（不升）
        nums = [-num for num in nums]
        return self.definitely_not_reduce(nums)


class LcsLis:
    def __init__(self):
        return

    def longest_common_subsequence(self, s1, s2) -> int:
        # 使用LIS的办法求LCS
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m-1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])

        return self.longest_increasing_subsequence(nums)

    @staticmethod
    def longest_increasing_subsequence(nums: List[int]) -> int:
        # 使用贪心二分求LIS
        stack = []
        for x in nums:
            idx = bisect_left(stack, x)
            if idx < len(stack):
                stack[idx] = x
            else:
                stack.append(x)
        # 还可以返回stack获得最长公共子序列
        return len(stack)

    def longest_common_subsequence_stack(self, s1, s2) -> List[int]:
        # 使用LIS的办法求LCS
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        m = len(s2)
        mapper = defaultdict(list)
        for i in range(m - 1, -1, -1):
            mapper[s2[i]].append(i)
        nums = []
        for c in s1:
            if c in mapper:
                nums.extend(mapper[c])

        res = self.longest_increasing_subsequence_stack(nums)
        return res

    @staticmethod
    def longest_increasing_subsequence_stack(nums: List[int]) -> List[int]:
        # 使用贪心二分求LIS
        n = len(nums)
        tops = [nums[0]]
        piles = [0] * n
        piles[0] = 0

        for i in range(1, n):
            if nums[i] > tops[-1]:
                piles[i] = len(tops)
                tops.append(nums[i])
            else:
                j = bisect.bisect_left(tops, nums[i])
                piles[i] = j
                tops[j] = nums[i]

        lis = []
        j = len(tops) - 1
        for i in range(n - 1, -1, -1):
            if piles[i] == j:
                lis.append(nums[i])
                j -= 1
        lis.reverse() # 反转列表，输出字典序最小的方案
        # 还可以返回stack获得最长公共子序列
        return lis


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2111(arr: List[int], k: int) -> int:
        # 模板：最长不降子序列
        ans = 0
        for i in range(k):
            lst = arr[i::k]
            ans += len(lst)-LongestIncreasingSubsequence().definitely_not_reduce(lst)
        return ans

    @staticmethod
    def lc_p1020(ac=FastIO()):
        # 模板：根据 dilworth 最长不升子序列的长度与分成不降子序列的最小组数（最长上升子序列的长度）
        nums = ac.read_list_ints()
        lis = LongestIncreasingSubsequence()
        ac.st(lis.definitely_not_increase(nums))
        ac.st(lis.definitely_increase(nums))
        return

    @staticmethod
    def lg_1439(ac=FastIO()):
        # 模板：最长公共子序列求解使用哈希映射转换为最长上升子序列
        n = ac.read_int()
        nums = ac.read_list_ints()
        ind = [0]*(n+1)
        for i, num in enumerate(nums):
            ind[num] = i
        nums = [ind[x] for x in ac.read_list_ints()]
        ac.st(LongestIncreasingSubsequence().definitely_increase(nums))
        return

    @staticmethod
    def lg_p5939(ac=FastIO()):
        # 模板：旋转后转换为 LIS 问题
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums = [[x + y, y - x] for x, y in nums]
        nums.sort(key=lambda it: [it[0], -it[1]])
        dp = []
        for _, y in nums:
            i = bisect.bisect_left(dp, y)
            if 0 <= i < len(dp):
                dp[i] = y
            else:
                dp.append(y)
        ac.st(len(dp))
        return

    @staticmethod
    def lg_p5978(ac=FastIO()):
        # 模板：经典 LIS 变形问题，贪心枚举前半部分
        n, x = ac.read_ints()
        nums = ac.read_list_ints()
        # 预处理后缀部分的最长 LIS 序列
        post = [0] * (n + 1)
        dp = []
        for i in range(n - 1, -1, -1):
            j = bisect.bisect_left(dp, -nums[i])
            post[i] = j + 1
            if 0 <= j < len(dp):
                dp[j] = -nums[i]
            else:
                dp.append(-nums[i])

        # 贪心减少前缀值并维护最长子序列
        ans = max(post)
        dp = []
        for i in range(n):
            j = bisect.bisect_left(dp, nums[i])
            ans = ac.max(ans, j + post[i])
            j = bisect.bisect_left(dp, nums[i] - x)
            if 0 <= j < len(dp):
                dp[j] = nums[i] - x
            else:
                dp.append(nums[i] - x)
        ac.st(ans)
        return

    @staticmethod
    def lg_p7957(ac=FastIO()):
        # 模板：经典 LMS 逆问题构造
        n, k = ac.read_ints()
        if k * k < n:
            ac.st(-1)
            return
        ans = []
        x = 1
        while len(ans) < n:
            rest = ac.min(n-len(ans), k)
            for y in range(x+rest-1, x-1, -1):
                ans.append(y)
            x = x+rest
        ac.lst(ans)
        return

    @staticmethod
    def lc_673(nums: List[int]) -> int:
        # 模板：经典求 LIS 子序列的个数 O(nlogn) 做法
        dp = []  # 维护 LIS 数组
        s = []  # 长度对应的方案和
        q = []  # 长度对应的末尾值与个数
        for num in nums:
            if not dp or num > dp[-1]:
                dp.append(num)
                length = len(dp)
            else:
                i = bisect.bisect_left(dp, num)
                dp[i] = num
                length = i + 1
            while len(s) <= len(dp):
                s.append(0)
            while len(q) <= len(dp):
                q.append(deque())

            if length == 1:
                s[length] += 1
                q[length].append([num, 1])
            else:
                # 使用队列与计数加和维护
                while q[length - 1] and q[length - 1][0][0] >= num:
                    s[length - 1] -= q[length - 1].popleft()[1]
                s[length] += s[length - 1]
                q[length].append([num, s[length - 1]])
        # 可以进一步变换求非严格递增子序列的个数
        return s[-1]

    @staticmethod
    def lc_1092(self, str1: str, str2: str) -> str:
        # 模板：经典利用LIS求LCS的最短公共超序列
        if len(str1) > len(str2):
            str1, str2 = str2, str1
        lcs_lis = LcsLis().longest_common_subsequence_stack(str1, str2)
        i = j = 0
        ans = ""
        for ind in lcs_lis:
            w = str2[ind]
            while str1[i] != w:
                ans += str1[i]
                i += 1
            while str2[j] != w:
                ans += str2[j]
                j += 1
            ans += w
            i += 1
            j += 1
        ans += str1[i:] + str2[j:]
        return ans


class TestGeneral(unittest.TestCase):

    def test_longest_increasing_subsequence(self):
        lis = LongestIncreasingSubsequence()
        nums = [1, 2, 3, 3, 2, 2, 1]
        assert lis.definitely_increase(nums) == 3
        assert lis.definitely_not_reduce(nums) == 4
        assert lis.definitely_reduce(nums) == 3
        assert lis.definitely_not_increase(nums) == 5
        return


if __name__ == '__main__':
    unittest.main()
