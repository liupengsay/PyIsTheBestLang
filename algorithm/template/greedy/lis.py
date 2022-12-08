"""
LIS：Longest Increasing Subsequence
问题1：最长单调递增子序列（上升）：<
问题2：最长单调不减子序列（不降）：<=
问题3：最长单调递减子序列（下降）：>
问题4：最长单调不增子序列（不升）：>=
对于数组来说，正数反可以将后两个问题3和4转换为前两个问题1和2进行解决
参考题目： P1020 [NOIP1999 普及组] 导弹拦截 https://www.luogu.com.cn/problem/P1020

"""

import bisect
nums = [int(w) for w in input().strip().split() if w]
n = len(nums)
dp = []
# 单调不增（取反后转化为不减问题）
for num in nums:
    i = bisect.bisect_right(dp,-num)
    if 0<=i<len(dp):
        dp[i] = -num
    else:
        dp.append(-num)
ans1 = len(dp)

# 单调递增
dp = []
for num in nums:
    i = bisect.bisect_left(dp, num)
    if 0 <= i < len(dp):
        dp[i] = num
    else:
        dp.append(num)

ans2 = len(dp)