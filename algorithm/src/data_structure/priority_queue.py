import unittest
from collections import deque
import random
from typing import List

from algorithm.src.basis.binary_search import BinarySearch
from algorithm.src.fast_io import FastIO

"""
算法：单调队列、双端队列
功能：维护单调性，计算滑动窗口最大值最小值
题目：

===================================力扣===================================
239. 滑动窗口最大值（https://leetcode.cn/problems/sliding-window-maximum/）滑动区间最大值
1696. 跳跃游戏 VI（https://leetcode.cn/problems/jump-game-vi/）经典优先队列 DP

===================================洛谷===================================
P2251 质量检测（https://www.luogu.com.cn/problem/P2251）滑动区间最小值
P2032 扫描（https://www.luogu.com.cn/problem/P2032）滑动区间最大值
P1750 出栈序列（https://www.luogu.com.cn/problem/P1750）经典题目，滑动指针窗口栈加队列
P2311 loidc，想想看（https://www.luogu.com.cn/problem/P2311）不定长滑动窗口最大值索引
P7175 [COCI2014-2015#4] PŠENICA（https://www.luogu.com.cn/problem/P7175）使用有序优先队列进行模拟
P7793 [COCI2014-2015#7] ACM（https://www.luogu.com.cn/problem/P7793）双端单调队列，进行最小值计算
P2216 [HAOI2007]理想的正方形（https://www.luogu.com.cn/problem/P2216）二维区间的滑动窗口最大最小值
P1886 滑动窗口 /【模板】单调队列（https://www.luogu.com.cn/problem/P1886）计算滑动窗口的最大值与最小值
P1725 琪露诺（https://www.luogu.com.cn/problem/P1725）单调队列和指针维护滑动窗口最大值加线性DP
P2827 [NOIP2016 提高组] 蚯蚓（https://www.luogu.com.cn/problem/P2827）经典单调队列
P3800 Power收集（https://www.luogu.com.cn/problem/P3800）单调队列优化矩阵DP
P1016 [NOIP1999 提高组] 旅行家的预算（https://www.luogu.com.cn/problem/P1016）单调队列，贪心模拟油箱，还可以增加每个站的油量限制
P1714 切蛋糕（https://www.luogu.com.cn/problem/P1714）前缀和加滑动窗口最小值，单调队列计算小于一定长度的最大连续子段和
P2629 好消息，坏消息（https://www.luogu.com.cn/problem/P2629）环形数组前缀和与滑动窗口最小值
P3522 [POI2011]TEM-Temperature（https://www.luogu.com.cn/problem/P3522）看不懂的队列与单调栈思想
P3957 [NOIP2017 普及组] 跳房子（https://www.luogu.com.cn/problem/P3957）二分加优先队列加DP
P4085 [USACO17DEC]Haybale Feast G（https://www.luogu.com.cn/problem/P4085）双指针加优先队列滑动窗口最小值
P4392 [BOI2007]Sound 静音问题（https://www.luogu.com.cn/problem/P4392）单调队列计算滑动窗口最大值

===================================AcWing=====================================
133. 蚯蚓（https://www.acwing.com/problem/content/135/）三个优先队列加一个偏移量
135. 最大子序和（https://www.acwing.com/problem/content/137/）双端队列计算不超过一定长度的最大子段和

参考：OI WiKi（xx）
"""


class PriorityQueue:
    def __init__(self):
        return

    @staticmethod
    def sliding_window(nums: List[int], k: int, method="max") -> List[int]:
        assert k >= 1
        # 模板: 计算滑动窗口最大值与最小值
        if method == "min":
            nums = [-num for num in nums]
        n = len(nums)
        stack = deque()
        ans = []
        for i in range(n):
            while stack and stack[0][1] <= i - k:
                stack.popleft()
            while stack and stack[-1][0] <= nums[i]:
                stack.pop()
            stack.append([nums[i], i])
            if i >= k - 1:
                ans.append(stack[0][0])
        if method == "min":
            ans = [-num for num in ans]
        return ans

    @staticmethod
    def sliding_window_all(nums: List[int], k: int, method="max") -> List[int]:
        assert k >= 1
        # 模板: 计算滑动窗口最大值与最小值
        if method == "min":
            nums = [-num for num in nums]
        n = len(nums)
        stack = deque()
        ans = []
        for i in range(n):
            while stack and stack[0][1] <= i - k:
                stack.popleft()
            while stack and stack[-1][0] <= nums[i]:
                stack.pop()
            stack.append([nums[i], i])
            ans.append(stack[0][0])
        if method == "min":
            ans = [-num for num in ans]
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1725(ac=FastIO()):

        # 模板：单调队列和指针维护滑动窗口最大值加线性DP
        inf = float("-inf")
        n, low, high = ac.read_ints()
        n += 1
        nums = ac.read_list_ints()
        dp = [-inf] * n
        dp[0] = nums[0]
        j = 0
        stack = deque()
        for i in range(1, n):
            while stack and stack[0][0] < i - high:
                stack.popleft()
            while j < n and j <= i - low:
                while stack and stack[-1][1] <= dp[j]:
                    stack.pop()
                stack.append([j, dp[j]])
                j += 1
            if stack:
                dp[i] = stack[0][1] + nums[i]
        ans = max(dp[x] for x in range(n) if x + high >= n)
        ac.st(ans)
        return

    @staticmethod
    def lc_239(self, nums: List[int], k: int) -> List[int]:
        # 模板：滑动窗口最大值
        return PriorityQueue().sliding_window(nums, k)

    @staticmethod
    def lg_p2032(ac=FastIO()):
        # 模板：滑动窗口最大值
        n, k = ac.read_ints()
        nums = ac.read_list_ints()
        ans = PriorityQueue().sliding_window(nums, k)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p2251(ac=FastIO()):
        # 模板：滑动窗口最小值
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        ans = PriorityQueue().sliding_window(nums, m, "min")
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p2216(ac=FastIO()):

        # 模板：二维滑动窗口最大值与滑动窗口最小值
        m, n, k = ac.read_ints()
        grid = [ac.read_list_ints() for _ in range(m)]

        ceil = [[0]*n for _ in range(m)]
        floor = [[0]*n for _ in range(m)]
        pq = PriorityQueue()
        for i in range(m):
            ceil[i] = pq.sliding_window_all(grid[i], k, "max")
            floor[i] = pq.sliding_window_all(grid[i], k, "min")
        for j in range(n):
            lst = pq.sliding_window_all([ceil[i][j] for i in range(m)], k, "max")
            for i in range(m):
                ceil[i][j] = lst[i]
            lst = pq.sliding_window_all([floor[i][j] for i in range(m)], k, "min")
            for i in range(m):
                floor[i][j] = lst[i]
        ans = ceil[k-1][k-1] - floor[k-1][k-1]
        for i in range(k-1, m):
            for j in range(k-1, n):
                ans = ac.min(ans, ceil[i][j]-floor[i][j])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1886(ac=FastIO()):
        # 模板：计算滑动窗口最大最小值
        n, k = ac.read_ints()
        nums = ac.read_list_ints()
        ans1 = []
        ans2 = []
        ceil = deque()
        floor = deque()
        for i in range(n):
            while ceil and ceil[0] < i-k+1:
                ceil.popleft()
            while ceil and nums[ceil[-1]] <= nums[i]:
                ceil.pop()
            ceil.append(i)

            while floor and floor[0] < i-k+1:
                floor.popleft()
            while floor and nums[floor[-1]] >= nums[i]:
                floor.pop()
            floor.append(i)

            if i >= k-1:
                ans1.append(nums[floor[0]])
                ans2.append(nums[ceil[0]])
        ac.lst(ans1)
        ac.lst(ans2)
        return

    @staticmethod
    def lg_p3800(ac=FastIO()):
        # 模板：单调队列优化矩阵DP
        m, n, k, t = ac.read_ints()
        dct = [dict() for _ in range(m)]
        for _ in range(k):
            x, y, val = ac.read_ints()
            x -= 1
            y -= 1
            dct[x][y] = val

        dp = [[0]*n for _ in range(2)]
        pre = 0
        for i in range(m):
            cur = 1-pre
            stack = deque()
            ind = 0
            for j in range(n):
                while stack and stack[0][0] < j-t:
                    stack.popleft()
                while ind < n and ind <= j+t:
                    while stack and stack[-1][1] <= dp[pre][ind]:
                        stack.pop()
                    stack.append([ind, dp[pre][ind]])
                    ind += 1
                dp[cur][j] = dct[i].get(j, 0) + stack[0][1]
            pre = cur
        ac.st(max(dp[pre]))
        return

    @staticmethod
    def ac_133(ac=FastIO()):
        # 模板：三个优先队列加一个偏移量
        n, m, q, u, v, t = ac.read_ints()
        nums1 = ac.read_list_ints()
        nums1 = deque(sorted(nums1, reverse=True))
        nums2 = deque()
        nums3 = deque()
        delta = 0
        ans1 = []
        ans2 = []
        for i in range(1, m+1):
            a = nums1[0] + delta if nums1 else -inf
            b = nums2[0] + delta if nums2 else -inf
            c = nums3[0] + delta if nums3 else -inf
            if a >= b and a >= c:
                x = a
                nums1.popleft()
                x1 = x*u//v
                nums2.append(x1-delta-q)
                nums3.append(x-x1-delta-q)
            elif b >= c and b >= a:
                x = b
                nums2.popleft()
                x1 = x*u//v
                nums2.append(x1-delta-q)
                nums3.append(x-x1-delta-q)
            else:
                x = c
                nums3.popleft()
                x1 = x*u//v
                nums2.append(x1-delta-q)
                nums3.append(x-x1-delta-q)
            delta += q
            if i % t == 0:
                ans1.append(x)

        ind = 0
        while nums1 or nums2 or nums3:
            a = nums1[0] + delta if nums1 else -inf
            b = nums2[0] + delta if nums2 else -inf
            c = nums3[0] + delta if nums3 else -inf
            if a >= b and a >= c:
                x = a
                nums1.popleft()
            elif b >= c and b >= a:
                x = b
                nums2.popleft()
            else:
                x = c
                nums3.popleft()
            ind += 1
            if ind % t == 0:
                ans2.append(x)
        ac.lst(ans1)
        ac.lst(ans2)
        return

    @staticmethod
    def lg_p1016(ac=FastIO()):
        # 模板：单调队列，贪心模拟油箱，还可以增加每个站的油量限制
        d1, c, d2, p, n = ac.read_floats()
        n = int(n)
        nums = [[0, p]] + [ac.read_list_floats() for _ in range(n)] + [[d1, 0]]
        nums.sort()
        stack = deque([[p, c]])  # 价格与油量
        ans = 0
        in_stack = c
        n = len(nums)
        for i in range(1, n):

            # 当前油箱的最大可行驶距离
            dis = nums[i][0]-nums[i-1][0]
            if in_stack*d2 < dis:
                ac.st("No Solution")
                return

            while dis:
                # 依次取出价格最低的油进行消耗
                x = ac.min(dis/d2, stack[0][1])
                ans += x*stack[0][0]
                dis -= x*d2
                stack[0][1] -= x
                in_stack -= x
                if not stack[0][1]:
                    stack.popleft()

            # 在当前站点补充更加便宜的油
            cur_p = nums[i][1]
            while stack and stack[-1][0] >= cur_p:
                in_stack -= stack.pop()[1]
            stack.append([cur_p, c - in_stack])
            in_stack = c
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1714(ac=FastIO()):
        # 模板：单调队列计算小于一定长度的最大连续子段和
        n, m = ac.read_ints()
        nums = ac.read_list_ints()
        ans = max(nums)
        pre = 0
        stack = deque([[-1, 0]])
        for i in range(n):
            pre += nums[i]
            # 滑动窗口记录最小值
            while stack and stack[0][0] <= i - m - 1:
                stack.popleft()
            while stack and stack[-1][1] >= pre:
                stack.pop()
            stack.append([i, pre])
            if stack:
                ans = ac.max(ans, pre - stack[0][1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p2629(ac=FastIO()):
        # 模板：环形数组前缀和与滑动窗口最小值
        n = ac.read_int()
        nums = ac.read_list_ints()
        nums = [0] + nums + nums
        ans = 0
        stack = deque([0])
        for i in range(1, n * 2):
            nums[i] += nums[i - 1]
            while stack and stack[0] <= i - n:
                stack.popleft()
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            stack.append(i)
            if i >= n:
                if nums[stack[0]] >= nums[i - n]:
                    ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p3957(ac=FastIO()):
        # 模板：二分加单调队列
        n, d, k = ac.read_ints()
        dis = [0]
        score = [0]
        for _ in range(n):
            x, s = ac.read_ints()
            dis.append(x)
            score.append(s)
        n += 1

        def check(g):
            dp = [-inf] * n
            stack = deque()
            dp[0] = score[0]
            floor = ac.max(1, d - g)
            ceil = d + g
            j = 0
            for i in range(1, n):
                # 注意此时使用双指针移动窗口
                while stack and stack[0][1] < dis[i] - ceil:
                    stack.popleft()
                while j < n and dis[i] - dis[j] >= floor:
                    if dis[i] - dis[j] > ceil:
                        j += 1
                        continue
                    while stack and stack[-1][0] <= dp[j]:
                        stack.pop()
                    stack.append([dp[j], dis[j]])
                    j += 1
                if stack:
                    dp[i] = stack[0][0] + score[i]
                    if dp[i] >= k:
                        return True
            return False

        ans = BinarySearch().find_int_left(0, dis[-1], check)
        ac.st(ans if check(ans) else -1)
        return

    @staticmethod
    def lg_p4085(ac=FastIO()):

        # 模板：双指针加优先队列滑动窗口最小值
        n, m = ac.read_ints()
        f = []
        s = []
        for i in range(n):
            a, b = ac.read_ints()
            f.append(a)
            s.append(b)

        # 注意指针与窗口的变动
        ans = inf
        stack = deque([])
        j = pre = 0
        for i in range(n):
            while stack and stack[0][0] < i:
                stack.popleft()
            while j < n and pre < m:
                pre += f[j]
                while stack and stack[-1][1] <= s[j]:
                    stack.pop()
                stack.append([j, s[j]])
                j += 1
            if pre >= m:
                ans = ac.min(ans, stack[0][1])
            pre -= f[i]
        ac.st(ans)
        return

    @staticmethod
    def lg_p4392(ac=FastIO()):
        # 模板：单调队列计算滑动窗口最大值
        n, m, c = ac.read_ints()
        ceil = deque()
        floor = deque()
        nums = ac.read_list_ints()
        ans = False
        for i in range(n):

            while ceil and ceil[0] <= i - m:
                ceil.popleft()
            while floor and floor[0] <= i - m:
                floor.popleft()

            while ceil and nums[ceil[-1]] <= nums[i]:
                ceil.pop()
            ceil.append(i)

            while floor and nums[floor[-1]] >= nums[i]:
                floor.pop()
            floor.append(i)

            if i >= m - 1 and nums[ceil[0]] - nums[floor[0]] <= c:
                ac.st(i - m + 2)
                ans = True
        if not ans:
            ac.st("NONE")
        return


class TestGeneral(unittest.TestCase):

    def test_priority_queue(self):
        pq = PriorityQueue()

        for _ in range(10):
            n = random.randint(100, 1000)
            nums = [random.randint(1, n) for _ in range(n)]
            k = random.randint(1, n)
            ans = pq.sliding_window(nums, k, "max")
            for i in range(n-k+1):
                assert ans[i] == max(nums[i:i+k])

            ans = pq.sliding_window(nums, k, "min")
            for i in range(n - k + 1):
                assert ans[i] == min(nums[i:i + k])
        return


if __name__ == '__main__':
    unittest.main()
