"""
Algorithm：单调队列、双端队列
Function：维护单调性，sliding_window最大值最小值

====================================LeetCode====================================
239（https://leetcode.com/problems/sliding-window-maximum/）滑动区间最大值
1696（https://leetcode.com/problems/jump-game-vi/）priority_queue DP
862（https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/description/）prefix_sum|单调双端队列DP
1425（https://leetcode.com/problems/constrained-subsequence-sum/）单调队列DP模板题

=====================================LuoGu======================================
2251（https://www.luogu.com.cn/problem/P2251）滑动区间最小值
2032（https://www.luogu.com.cn/problem/P2032）滑动区间最大值
1750（https://www.luogu.com.cn/problem/P1750）题目，滑动pointer窗口栈|队列
2311（https://www.luogu.com.cn/problem/P2311）不定长sliding_window最大值索引
7175（https://www.luogu.com.cn/problem/P7175）有序priority_queueimplemention
7793（https://www.luogu.com.cn/problem/P7793）双端单调队列，最小值
2216（https://www.luogu.com.cn/problem/P2216）二维区间的sliding_window最大最小值
1886（https://www.luogu.com.cn/problem/P1886）sliding_window的最大值与最小值
1725（https://www.luogu.com.cn/problem/P1725）单调队列和pointer维护sliding_window最大值|线性DP
2827（https://www.luogu.com.cn/problem/P2827）单调队列
3800（https://www.luogu.com.cn/problem/P3800）单调队列优化matrix_dp
1016（https://www.luogu.com.cn/problem/P1016）单调队列，greedyimplemention油箱，还可以增|每个站的油量限制
1714（https://www.luogu.com.cn/problem/P1714）prefix_sum|sliding_window最小值，单调队列小于一定长度的最大连续子段和
2629（https://www.luogu.com.cn/problem/P2629）环形数组prefix_sum与sliding_window最小值
3522（https://www.luogu.com.cn/problem/P3522）看不懂的队列与monotonic_stack|思想
3957（https://www.luogu.com.cn/problem/P3957）binary_search|priority_queue|DP
4085（https://www.luogu.com.cn/problem/P4085）two_pointer|priority_queuesliding_window最小值
4392（https://www.luogu.com.cn/problem/P4392）单调队列sliding_window最大值

=====================================AcWing=====================================
133（https://www.acwing.com/problem/content/135/）三个priority_queue|一个偏移量
135（https://www.acwing.com/problem/content/137/）双端队列不超过一定长度的最大子段和

"""
from collections import deque
from math import inf
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.data_structure.priority_queue.template import PriorityQueue
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1725(ac=FastIO()):
        # 单调队列和pointer维护sliding_window最大值|线性DP
        n, low, high = ac.read_list_ints()
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
    def lc_239(nums: List[int], k: int) -> List[int]:
        # sliding_window最大值
        return PriorityQueue().sliding_window(nums, k)

    @staticmethod
    def lc_862(nums: List[int], k: int) -> int:
        # prefix_sum|单调双端队列DP
        n = len(nums)
        stack = deque([0])
        ind = deque([-1])
        pre = 0
        ans = n + 1
        for i in range(n):
            pre += nums[i]
            while stack and stack[0] <= pre - k:
                stack.popleft()
                j = ind.popleft()
                if i - j < ans:
                    ans = i - j
            while stack and stack[-1] >= pre:
                stack.pop()
                ind.pop()
            stack.append(pre)
            ind.append(i)
        return ans if ans < n + 1 else -1

    @staticmethod
    def lg_p2032(ac=FastIO()):
        # sliding_window最大值
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        ans = PriorityQueue().sliding_window(nums, k)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p2251(ac=FastIO()):
        # sliding_window最小值
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        ans = PriorityQueue().sliding_window(nums, m, "min")
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p2216(ac=FastIO()):

        # 二维sliding_window最大值与sliding_window最小值
        m, n, k = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]

        ceil = [[0] * n for _ in range(m)]
        floor = [[0] * n for _ in range(m)]
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
        ans = ceil[k - 1][k - 1] - floor[k - 1][k - 1]
        for i in range(k - 1, m):
            for j in range(k - 1, n):
                ans = ac.min(ans, ceil[i][j] - floor[i][j])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1886(ac=FastIO()):
        # sliding_window最大最小值
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        ans1 = []
        ans2 = []
        ceil = deque()
        floor = deque()
        for i in range(n):
            while ceil and ceil[0] < i - k + 1:
                ceil.popleft()
            while ceil and nums[ceil[-1]] <= nums[i]:
                ceil.pop()
            ceil.append(i)

            while floor and floor[0] < i - k + 1:
                floor.popleft()
            while floor and nums[floor[-1]] >= nums[i]:
                floor.pop()
            floor.append(i)

            if i >= k - 1:
                ans1.append(nums[floor[0]])
                ans2.append(nums[ceil[0]])
        ac.lst(ans1)
        ac.lst(ans2)
        return

    @staticmethod
    def lg_p3800(ac=FastIO()):
        # 单调队列优化matrix_dp
        m, n, k, t = ac.read_list_ints()
        dct = [dict() for _ in range(m)]
        for _ in range(k):
            x, y, val = ac.read_list_ints()
            x -= 1
            y -= 1
            dct[x][y] = val

        dp = [[0] * n for _ in range(2)]
        pre = 0
        for i in range(m):
            cur = 1 - pre
            stack = deque()
            ind = 0
            for j in range(n):
                while stack and stack[0][0] < j - t:
                    stack.popleft()
                while ind < n and ind <= j + t:
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
        # 三个priority_queue|一个偏移量
        n, m, q, u, v, t = ac.read_list_ints()
        nums1 = ac.read_list_ints()
        nums1 = deque(sorted(nums1, reverse=True))
        nums2 = deque()
        nums3 = deque()
        delta = 0
        ans1 = []
        ans2 = []
        for i in range(1, m + 1):
            a = nums1[0] + delta if nums1 else -inf
            b = nums2[0] + delta if nums2 else -inf
            c = nums3[0] + delta if nums3 else -inf
            if a >= b and a >= c:
                x = a
                nums1.popleft()
                x1 = x * u // v
                nums2.append(x1 - delta - q)
                nums3.append(x - x1 - delta - q)
            elif b >= c and b >= a:
                x = b
                nums2.popleft()
                x1 = x * u // v
                nums2.append(x1 - delta - q)
                nums3.append(x - x1 - delta - q)
            else:
                x = c
                nums3.popleft()
                x1 = x * u // v
                nums2.append(x1 - delta - q)
                nums3.append(x - x1 - delta - q)
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
        # 单调队列，greedyimplemention油箱，还可以增|每个站的油量限制
        d1, c, d2, p, n = ac.read_list_floats()
        n = int(n)
        nums = [[0, p]] + [ac.read_list_floats() for _ in range(n)] + [[d1, 0]]
        nums.sort()
        stack = deque([[p, c]])  # 价格与油量
        ans = 0
        in_stack = c
        n = len(nums)
        for i in range(1, n):

            # 当前油箱的最大可行驶距离
            dis = nums[i][0] - nums[i - 1][0]
            if in_stack * d2 < dis:
                ac.st("No Solution")
                return

            while dis:
                # 依次取出价格最低的油消耗
                x = ac.min(dis / d2, stack[0][1])
                ans += x * stack[0][0]
                dis -= x * d2
                stack[0][1] -= x
                in_stack -= x
                if not stack[0][1]:
                    stack.popleft()

            # 在当前站点补充更|便宜的油
            cur_p = nums[i][1]
            while stack and stack[-1][0] >= cur_p:
                in_stack -= stack.pop()[1]
            stack.append([cur_p, c - in_stack])
            in_stack = c
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1714(ac=FastIO()):
        # 单调队列小于一定长度的最大连续子段和
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        ans = max(nums)
        pre = 0
        stack = deque([[-1, 0]])
        for i in range(n):
            pre += nums[i]
            # sliding_window记录最小值
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
        # 环形数组prefix_sum与sliding_window最小值
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
        # binary_search|单调队列
        n, d, k = ac.read_list_ints()
        dis = [0]
        score = [0]
        for _ in range(n):
            x, s = ac.read_list_ints()
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
                # 注意此时two_pointer移动窗口
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

        # two_pointer|priority_queuesliding_window最小值
        n, m = ac.read_list_ints()
        f = []
        s = []
        for i in range(n):
            a, b = ac.read_list_ints()
            f.append(a)
            s.append(b)

        # 注意pointer与窗口的变动
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
        # 单调队列sliding_window最大值
        n, m, c = ac.read_list_ints()
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