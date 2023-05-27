import random
import unittest
from collections import deque
from typing import List

from sortedcontainers import SortedList
from algorithm.src.fast_io import FastIO

"""

算法：使用数组作为链表维护前驱后驱
功能：

题目：xx（xx）

===================================力扣===================================
2617. 网格图中最少访问的格子数（https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/）使用数组维护链表的前后节点信息
2612. 最少翻转操作数（https://leetcode.cn/problems/minimum-reverse-operations/）使用数组维护链表的前后节点信息

===================================牛客===================================
牛牛排队伍（https://ac.nowcoder.com/acm/contest/49888/C）使用数组维护链表的前后节点信息

===================================洛谷===================================
P5462 X龙珠（https://www.luogu.com.cn/problem/P5462）经典使用双向链表贪心选取最大字典序队列

================================CodeForces================================
E. Two Teams（https://codeforces.com/contest/1154/problem/E）使用数组维护链表的前后节点信息

================================AcWing===================================
136. 邻值查找（https://www.acwing.com/problem/content/138/）链表逆序删除，查找前后最接近的值


参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1154e(ac=FastIO()):

        # 模板：使用链表维护前后的节点信息
        n, k = ac.read_ints()
        nums = ac.read_list_ints()
        ans = [0] * n
        pre = [i - 1 for i in range(n)]
        nex = [i + 1 for i in range(n)]
        ind = [0] * n
        for i in range(n):
            ind[nums[i] - 1] = i

        step = 1
        for num in range(n - 1, -1, -1):
            i = ind[num]
            if ans[i]:
                continue
            ans[i] = step
            left, right = pre[i], nex[i]
            for _ in range(k):
                if left != -1:
                    ans[left] = step
                    left = pre[left]
                else:
                    break
            for _ in range(k):
                if right != n:
                    ans[right] = step
                    right = nex[right]
                else:
                    break
            if left >= 0:
                nex[left] = right
            if right < n:
                pre[right] = left
            step = 3 - step
        ac.st("".join(str(x) for x in ans))
        return

    @staticmethod
    def nc_247577(ac=FastIO()):
        # 模板：使用链表维护前后的节点信息
        n, k = ac.read_ints()
        pre = list(range(-1, n + 1))
        post = list(range(1, n + 3))
        assert len(pre) == len(post) == n + 2
        for _ in range(k):
            op, x = ac.read_list_ints()
            if op == 1:
                a = pre[x]
                b = post[x]
                if 1 <= b <= n:
                    pre[b] = a
                if 1 <= a <= n:
                    post[a] = b
                pre[x] = post[x] = -1
            else:
                ac.st(pre[x])
        return
    
    @staticmethod
    def lc_2617(grid: List[List[int]]) -> int:
        # 模板：使用链表维护前后的节点信息

        m, n = len(grid), len(grid[0])
        inf = inf
        dis = [[inf] * n for _ in range(m)]
        row_nex = [list(range(1, n + 1)) for _ in range(m)]
        row_pre = [list(range(-1, n - 1)) for _ in range(m)]
        col_nex = [list(range(1, m + 1)) for _ in range(n)]
        col_pre = [list(range(-1, m - 1)) for _ in range(n)]
        stack = deque([[0, 0]])
        dis[0][0] = 1

        while stack:
            i, j = stack.popleft()
            d = dis[i][j]
            x = grid[i][j]
            if x == 0:
                continue

            # 按照行取出可以访问到的节点
            pre = row_pre[i]
            nex = row_nex[i]
            y = nex[j]
            lst = []
            while y <= j + x and 0 <= y < n:
                if dis[i][y] == inf:
                    dis[i][y] = d + 1
                    if i == m - 1 and y == n - 1:
                        return d + 1
                    stack.append([i, y])
                lst.append(y)
                y = nex[y]
            # 更新前驱后驱
            for w in lst:
                nex[w] = y
                pre[w] = pre[j]

            # 按照列取出可以访问到的节点
            pre = col_pre[j]
            nex = col_nex[j]
            y = nex[i]
            lst = []
            while y <= i + x and 0 <= y < m:
                if dis[y][j] == inf:
                    dis[y][j] = d + 1
                    if y == m - 1 and j == n - 1:
                        return d + 1
                    stack.append([y, j])
                lst.append(y)
                y = nex[y]
            # 更新前驱后驱
            for w in lst:
                nex[w] = y
                pre[w] = pre[i]

        ans = dis[-1][-1]
        return ans if ans < inf else -1

    @staticmethod
    def ac_136(ac=FastIO()):
        # 模板：链表逆序删除，查找前后最接近的值，也可直接使用SortedList
        n = ac.read_int()
        nums = ac.read_list_ints()
        ind = list(range(n))
        ind.sort(key=lambda it: nums[it])
        dct = {nums[i]: i for i in range(n)}
        pre = [-1]*n
        post = [-1]*n
        for i in range(1, n):
            a, b = ind[i-1], ind[i]
            post[a] = b
            pre[b] = a
        ans = []
        for x in range(n-1, 0, -1):
            num = nums[x]
            i = dct[num]
            a = pre[i]
            b = post[i]
            if a != -1 and b != -1:
                if abs(num-nums[a]) < abs(num-nums[b]) or (abs(num-nums[a]) == abs(num-nums[b]) and nums[a] < nums[b]):
                    ans.append([abs(num-nums[a]), a+1])
                else:
                    ans.append([abs(num - nums[b]), b + 1])
                post[a] = b
                pre[b] = a
            elif a != -1:
                ans.append([abs(num - nums[a]), a + 1])
                post[a] = post[i]
            else:
                ans.append([abs(num - nums[b]), b + 1])
                pre[b] = pre[i]
        for i in range(n-2, -1, -1):
            ac.lst(ans[i])
        return

    @staticmethod
    def lg_p5462(ac=FastIO()):
        # 模板：经典使用双向链表贪心选取最大字典序队列
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = [-1] * (n + 1)
        post = [-1] * (n + 1)
        for i in range(n):
            if i:
                pre[nums[i]] = nums[i - 1]
            if i + 1 < n:
                post[nums[i]] = nums[i + 1]

        # 从大到小取出
        visit = [0] * (n + 1)
        ans = []
        for num in range(n, 0, -1):
            if visit[num] or post[num] == -1:
                continue
            visit[num] = visit[post[num]] = 1
            ans.extend([num, post[num]])
            x, y = pre[num], post[post[num]]
            if x != -1:
                post[x] = y
            if y != -1:
                pre[y] = x
        ac.lst(ans)
        return


class TestGeneral(unittest.TestCase):

    def test_xx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()