"""
算法：堆（优先队列）、Huffman树（霍夫曼树）
功能：通常用于需要贪心的场景
题目：

===================================LeetCode===================================
630（https://leetcode.com/problems/course-schedule-iii/）用一个堆延迟选择贪心维护最优
2454（https://leetcode.com/problems/next-greater-element-iv/）使用两个堆维护下下个更大元素即出队两次时遇见的元素，也可以使用经典的哈希加SortedList
2402（https://leetcode.com/problems/meeting-rooms-iii/）使用两个堆模拟进行会议室安排并进行计数
2386（https://leetcode.com/problems/find-the-k-sum-of-an-array/）转换思路使用堆维护最大和第 K 次出队的则为目标结果
2163（https://leetcode.com/problems/minimum-difference-in-sums-after-removal-of-elements/）预处理前缀后缀最大最小的 K 个数和再进行枚举分割点
1792（https://leetcode.com/problems/maximum-average-pass-ratio/）贪心依次给增幅最大的班级人数加 1
295（https://leetcode.com/problems/find-median-from-data-stream/）用两个堆维护中位数
2542（https://leetcode.com/problems/maximum-subsequence-score/）贪心排序枚举加堆维护最大的k个数进行计算
2263（https://leetcode.com/problems/make-array-non-decreasing-or-non-increasing/）大根堆贪心使得序列非降的最小操作次数

===================================LuoGu==================================
1168（https://www.luogu.com.cn/problem/P1168） 用两个堆维护中位数
1801（https://www.luogu.com.cn/problem/P1801）用两个堆维护第K小
2085（https://www.luogu.com.cn/problem/P2085）用数学加一个堆维护前K小
1631（https://www.luogu.com.cn/problem/P1631）用一个堆和指针组合维护前K小
4053（https://www.luogu.com.cn/problem/P4053）用一个堆延迟选择贪心维护最优，经典课程表 III
1878（https://www.luogu.com.cn/problem/P1878）用哈希加一个堆进行模拟计算
3620（https://www.luogu.com.cn/problem/P3620）贪心思想加二叉堆与双向链表优
2168（https://www.luogu.com.cn/problem/P2168）霍夫曼树与二叉堆贪心
2278（https://www.luogu.com.cn/problem/P2278）使用二叉堆模拟CPU占用
1717（https://www.luogu.com.cn/problem/P1717）枚举最远到达地点进行二叉堆贪心选取
1905（https://www.luogu.com.cn/problem/P1905）二叉堆从大到小贪心摆放
2409（https://www.luogu.com.cn/problem/P2409）经典二叉堆，计算最小的k个和
2949（https://www.luogu.com.cn/problem/P2949）二叉堆贪心模拟懒惰延迟删除
6033（https://www.luogu.com.cn/problem/P6033）经典贪心升级版可用双端队列优化
4597（https://www.luogu.com.cn/problem/P4597）大根堆贪心使得序列非降的最小操作次数

===================================AcWing======================================
146（https://www.acwing.com/problem/content/description/148/）小顶堆计算经典问题m个数组最小的n个子序列和，同样可以计算最大的
147（https://www.acwing.com/problem/content/description/149/）贪心思想加二叉堆与双向链表优化
148（https://www.acwing.com/problem/content/150/）贪心二叉堆，霍夫曼树Huffman Tree的思想，每次优先合并较小的
149（https://www.acwing.com/problem/content/description/151/）霍夫曼树与二叉堆贪心



参考：OI WiKi（xx）
"""

import heapq
from collections import deque, defaultdict
from heapq import heappushpop, heappush, heappop
from math import inf
from typing import List

from sortedcontainers import SortedList

from src.data_structure.small_heapq.template import MedianFinder
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_2454_1(nums: List[int]) -> List[int]:
        # 模板：经典哈希排序加SortedList
        n = len(nums)
        dct = defaultdict(list)
        for i in range(n):
            dct[nums[i]].append(i)
        lst = SortedList()
        ans = [-1] * n
        for num in sorted(dct, reverse=True):
            for i in dct[num]:
                j = lst.bisect_left(i)
                if 0 <= j + 1 < len(lst):
                    ans[i] = nums[lst[j + 1]]
            for i in dct[num]:
                lst.add(i)
        return ans

    @staticmethod
    def lc_2454_2(nums: List[int]) -> List[int]:

        # 模板：经典单调栈加小顶堆
        n = len(nums)
        ans = [-1] * n
        mono_stack = []
        small_stack = []
        for i in range(n):
            while small_stack and small_stack[0][0] < nums[i]:
                ans[heapq.heappop(small_stack)[1]] = nums[i]

            while mono_stack and nums[mono_stack[-1]] < nums[i]:
                j = mono_stack.pop()
                heapq.heappush(small_stack, [nums[j], j])
            mono_stack.append(i)

        return ans

    @staticmethod
    def lg_1198(ac=FastIO()):
        # 模板：使用两个堆维护中位数
        n = ac.read_int()
        nums = ac.read_list_ints()
        arr = MedianFinder()
        for i in range(n):
            arr.add_num(nums[i])
            if i % 2 == 0:
                ac.st(arr.find_median())
        return

    @staticmethod
    def lc_1792(classes, extra_students):
        # 模板：使用堆进行贪心模拟每次选择最优
        stack = []
        for p, t in classes:
            heapq.heappush(stack, [p / t - (p + 1) / (t + 1), p, t])
        for _ in range(extra_students):
            r, p, t = heapq.heappop(stack)
            p += 1
            t += 1
            # 关键点在于优先级的设置为 p / t - (p + 1) / (t + 1)
            heapq.heappush(stack, [p / t - (p + 1) / (t + 1), p, t])
        return sum(p / t for _, p, t in stack) / len(classes)

    @staticmethod
    def lc_630(courses: List[List[int]]) -> int:
        # 模板：经典反悔堆，遍历过程选择使用更优的
        courses.sort(key=lambda x: x[1])
        # 按照结束时间排序
        stack = []
        day = 0
        for duration, last in courses:
            if day + duration <= last:
                day += duration
                heapq.heappush(stack, -duration)
            else:
                # 如果有学习时间更短的课程则进行替换
                if stack and -stack[0] > duration:
                    day += heapq.heappop(stack) + duration
                    heapq.heappush(stack, -duration)
        return len(stack)

    @staticmethod
    def ac_146(ac=FastIO()):
        # 模板：小顶堆计算经典问题m个数组最小的n个子序列和，同样可以计算最大的
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            grid = [sorted(ac.read_list_ints()) for _ in range(m)]
            grid = [g for g in grid if g]
            m = len(grid)

            pre = grid[0]
            for i in range(1, m):
                cur = grid[i][:]
                nex = []
                stack = [[pre[0] + cur[0], 0, 0]]
                dct = set()
                while stack and len(nex) < n:
                    val, i, j = heapq.heappop(stack)
                    if (i, j) in dct:
                        continue
                    dct.add((i, j))
                    nex.append(val)
                    if i + 1 < n:
                        heapq.heappush(stack, [pre[i + 1] + cur[j], i + 1, j])
                    if j + 1 < n:
                        heapq.heappush(stack, [pre[i] + cur[j + 1], i, j + 1])
                pre = nex[:]
            ac.lst(pre)
        return

    @staticmethod
    def ac_147(ac=FastIO()):
        # 模板：贪心思想加二叉堆与双向链表优化

        n, k = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]

        # 假如虚拟的头节点并初始化
        diff = [inf] + [nums[i + 1] - nums[i] for i in range(n - 1)] + [inf]
        stack = [[diff[i], i] for i in range(1, n)]
        heapq.heapify(stack)
        pre = [i - 1 for i in range(n + 1)]
        post = [i + 1 for i in range(n + 1)]
        pre[0] = 0
        post[n] = n

        # 记录删除过的点
        ans = 0
        delete = [0] * (n + 1)
        while k:
            val, i = heapq.heappop(stack)
            if delete[i]:
                continue
            ans += diff[i]

            # 加入新点删除旧点
            left = diff[pre[i]]
            right = diff[post[i]]
            new = left + right - diff[i]
            diff[i] = new
            delete[pre[i]] = 1
            delete[post[i]] = 1

            pre[i] = pre[pre[i]]
            post[pre[i]] = i

            post[i] = post[post[i]]
            pre[post[i]] = i
            heapq.heappush(stack, [new, i])
            k -= 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p2168(ac=FastIO()):
        # 模板：二叉堆贪心与霍夫曼树Huffman Tree
        n, k = ac.read_list_ints()
        stack = [[ac.read_int(), 0] for _ in range(n)]
        heapq.heapify(stack)
        while (len(stack) - 1) % (k - 1) != 0:
            heapq.heappush(stack, [0, 0])
        ans = 0
        while len(stack) > 1:
            cur = 0
            dep = 0
            for _ in range(k):
                val, d = heapq.heappop(stack)
                cur += val
                dep = ac.max(dep, d)
            ans += cur
            heapq.heappush(stack, [cur, dep + 1])
        ac.st(ans)
        ac.st(stack[0][1])
        return

    @staticmethod
    def lg_p1631(ac=FastIO()):
        # 模板：求两个数组的前 n 个最小的元素和
        n = ac.read_int()
        nums1 = ac.read_list_ints()
        nums2 = ac.read_list_ints()
        # 初始时加入所有第二个数组的索引位置
        stack = [[nums1[0] + nums2[j], 0, j] for j in range(n)]
        # 不重不漏枚举所有索引组合
        heapq.heapify(stack)
        ans = []
        for _ in range(n):
            val, i, j = heapq.heappop(stack)
            ans.append(val)
            if i + 1 < n:
                heapq.heappush(stack, [nums1[i + 1] + nums2[j], i + 1, j])
        ac.lst(ans)
        return

    @staticmethod
    def lg_p4053(ac=FastIO()):
        # 模板：懒惰删除，模拟贪心
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: it[1])
        pre = 0
        stack = []
        for a, b in nums:
            if pre + a <= b:
                heapq.heappush(stack, -a)
                pre += a
            else:
                if stack and -a > stack[0]:
                    pre += heapq.heappop(stack)
                    pre += a
                    heapq.heappush(stack, -a)
        ac.st(len(stack))
        return

    @staticmethod
    def lg_p2085(ac=FastIO()):
        # 模板：利用一元二次方程的单调性与指针堆优化进行贪心选取
        n, m = ac.read_list_ints()
        stack = []
        for _ in range(n):
            a, b, c = ac.read_list_ints()
            heapq.heappush(stack, [a + b + c, 1, a, b, c])
        ans = []
        while len(ans) < m:
            val, x, a, b, c = heapq.heappop(stack)
            ans.append(val)
            x += 1
            heapq.heappush(stack, [a * x * x + b * x + c, x, a, b, c])
        ac.lst(ans)
        return

    @staticmethod
    def lg_p2278(ac=FastIO()):
        # 模板：使用堆模拟应用
        now = []  # idx, reach, need, level, end
        ans = []
        stack = []  # -level, reach, need, idx
        pre = 0
        while True:
            lst = ac.read_list_ints()  # idx, reach, need, level
            if not lst:
                break

            # 当前达到时刻之前可以完成的任务消除掉
            while now and now[-1] <= lst[1]:
                ans.append([now[0], now[-1]])
                pre = now[-1]
                if stack:
                    level, reach, need, idx = heapq.heappop(stack)
                    now = [idx, reach, need, -level, ac.max(pre, reach) + need]
                else:
                    now = []

            # 取出还有的任务运行
            if not now and stack:
                level, reach, need, idx = heapq.heappop(stack)
                now = [idx, reach, need, -level, ac.max(pre, reach) + need]

            # 执行任务等级不低于当前任务，当前任务直接入队
            if now and now[3] >= lst[-1]:
                idx, reach, need, level = lst
                heapq.heappush(stack, [-level, reach, need, idx])
            elif now:
                # 当前任务等级更高，进行替换，注意剩余时间
                idx, reach, need, level, end = now
                heapq.heappush(stack, [-level, reach, end - lst[1], idx])
                idx, reach, need, level = lst
                now = [idx, reach, need, level, ac.max(pre, reach) + need]
            else:
                # 无执行任务，直接执行当前任务
                idx, reach, need, level = lst
                now = [idx, reach, need, level, ac.max(pre, reach) + need]

        while stack:
            # 执行剩余任务
            ans.append([now[0], now[-1]])
            pre = now[-1]
            level, reach, need, idx = heapq.heappop(stack)
            now = [idx, reach, need, -level, ac.max(pre, reach) + need]
        ans.append([now[0], now[-1]])
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def lg_p1717(ac=FastIO()):
        # 模板：枚举最远到达地点进行二叉堆贪心选取
        ans = 0
        n = ac.read_int()
        h = ac.read_int() * 60
        f = ac.read_list_ints()
        d = ac.read_list_ints()
        t = [0] + ac.read_list_ints()
        for i in range(n):
            tm = sum(t[:i + 1]) * 5
            stack = [[-f[j], j] for j in range(i + 1)]
            heapq.heapify(stack)
            cur = 0
            while tm + 5 <= h and stack:
                val, j = heapq.heappop(stack)
                val = -val
                cur += val
                tm += 5
                if val - d[j] > 0:
                    heapq.heappush(stack, [-val + d[j], j])
            ans = ac.max(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1905(ac=FastIO()):
        # 模板：二叉堆从大到小贪心摆放
        ac.read_int()
        p = ac.read_int()
        lst = ac.read_list_ints()
        ans = [[0] for _ in range(p)]
        stack = [[ans[i][0], i] for i in range(p)]
        lst.sort(reverse=True)
        for num in lst:
            d, i = heapq.heappop(stack)
            ans[i][0] += num
            ans[i].append(num)
            heapq.heappush(stack, [ans[i][0], i])
        for a in ans:
            ac.lst(a[1:])
        return

    @staticmethod
    def lg_p2409(ac=FastIO()):
        # 模板：经典二叉堆，计算最小的k个和
        n, k = ac.read_list_ints()
        pre = ac.read_list_ints()[1:]
        pre.sort()
        for _ in range(n - 1):
            cur = ac.read_list_ints()[1:]
            cur.sort()
            nex = []
            for x in cur:
                for num in pre:
                    if len(nex) == k and -num - x < nex[0]:
                        break
                    heapq.heappush(nex, -num - x)
                    if len(nex) > k:
                        heapq.heappop(nex)
            pre = sorted([-x for x in nex])
        ac.lst(pre[:k])
        return

    @staticmethod
    def lg_p2949(ac=FastIO()):
        # 模板：二叉堆贪心模拟懒惰延迟删除
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: it[0])
        ans = 0
        stack = []
        for d, p in nums:
            heapq.heappush(stack, p)
            ans += p
            if len(stack) > d:
                ans -= heapq.heappop(stack)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6033(ac=FastIO()):
        # 模板：经典队列 O(n) 模拟合并果子
        ac.read_int()
        pre = deque(sorted(ac.read_list_ints()))
        post = deque()
        ans = 0
        while len(pre) + len(post) > 1:
            if not pre:
                cur = post.popleft() + post.popleft()
                ans += cur
                post.append(cur)
                continue
            if not post:
                cur = pre.popleft() + pre.popleft()
                ans += cur
                post.append(cur)
                continue
            if pre[0] < post[0]:
                a = pre.popleft()
            else:
                a = post.popleft()
            if pre and (not post or pre[0] < post[0]):
                b = pre.popleft()
            else:
                b = post.popleft()
            ans += a + b
            post.append(a + b)
        ac.st(ans)
        return

    @staticmethod
    def lc_2263(nums: List[int]) -> int:

        def helper(lst: List[int]) -> int:
            # 模板：大根堆贪心使得序列非降的最小操作次数
            res, pq = 0, []  # 大根堆
            for num in lst:
                if not pq:
                    heappush(pq, -num)
                else:
                    pre = -pq[0]
                    if pre > num:
                        res += pre - num
                        heappushpop(pq, -num)
                    heappush(pq, -num)
            return res

        return min(helper(nums), helper(nums[::-1]))

    @staticmethod
    def lc_2386(nums: List[int], k: int) -> int:
        # 模板：转换思路使用堆维护最大和第 K 次出队的则为目标结果
        n = len(nums)
        tot = 0
        for i in range(n):
            if nums[i] >= 0:
                tot += nums[i]
            else:
                nums[i] = -nums[i]
        nums.sort()
        # 经典思维题，类似Dijkstra的思想从大到小枚举子序列的和
        stack = [[nums[0], 0]]
        for _ in range(k - 1):
            pre, i = heappop(stack)
            if i < n:
                heapq.heappush(stack, [pre + nums[i], i + 1])
                if i:
                    heapq.heappush(stack, [pre + nums[i] - nums[i - 1], i + 1])
        return -stack[0][0]