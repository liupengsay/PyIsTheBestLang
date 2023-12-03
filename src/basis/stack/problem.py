"""

算法：栈、RBS（合法括号子序列）
功能：模拟题中常见，如括号之类的，后进先出，升级版应用有单调栈、最大栈和最小栈
题目：

===================================LeetCode===================================
2197（https://leetcode.com/problems/replace-non-coprime-numbers-in-array/）结合数学使用栈进行模拟
394（https://leetcode.com/problems/decode-string/）经典解码带括号成倍的字符和数字
1096（https://leetcode.com/problems/brace-expansion-ii/）使用栈进行字符解码
2116（https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/）经典栈贪心匹配括号
857（https://leetcode.com/problems/minimum-cost-to-hire-k-workers/）经典贪心排序枚举，使用堆维护K个最小值的和
2542（https://leetcode.com/problems/maximum-subsequence-score/）经典排序后枚举使用堆维护K最大的和，类似LC857
2813（https://leetcode.com/problems/maximum-elegance-of-a-k-length-subsequence/）经典思维题排序后枚举，维护长度为K的子序列最大函数值
2462（https://leetcode.com/problems/total-cost-to-hire-k-workers/）使用堆进行贪心模拟
1705（https://leetcode.com/problems/maximum-number-of-eaten-apples/）使用堆进行贪心模拟
1750（https://leetcode.com/problems/minimum-length-of-string-after-deleting-similar-ends/description/）经典栈模拟
2296（https://leetcode.com/problems/design-a-text-editor/description/）经典左右两个栈进行模拟

===================================LuoGu==================================
1944（https://www.luogu.com.cn/problem/P1944）最长连续合法括号字串长度
2201（https://www.luogu.com.cn/problem/P2201）双栈模拟指针移动同时记录前缀和与前序最大前缀和
4387（https://www.luogu.com.cn/problem/P4387）模拟入栈出栈队列判断是否可行
7674（https://www.luogu.com.cn/problem/P7674）使用栈模仿消除
3719（https://www.luogu.com.cn/problem/P3719）字符串运算展开
1974（https://www.luogu.com.cn/problem/P1974）贪心队列模拟
3551（https://www.luogu.com.cn/problem/P3551）栈与计数指针
3719（https://www.luogu.com.cn/problem/P3719）栈模拟

================================CodeForces================================
C. Longest Regular Bracket Sequence（https://codeforces.com/problemset/problem/5/C）最长连续合法括号子序列以及个数
E. Almost Regular Bracket Sequence（https://codeforces.com/problemset/problem/1095/E）计算改变一个括号后是的字符串合法的位置数

================================AtCoder================================
D - 3N Numbers（https://atcoder.jp/contests/abc062/tasks/arc074_b）经典堆与前后缀结合


================================AcWing===================================
128（https://www.acwing.com/problem/content/130/）堆栈模拟
129（https://www.acwing.com/problem/content/131/）经典卡特兰数，栈模拟判定出栈入栈合法性
132（https://www.acwing.com/problem/content/134/）双端队列依次出队入队
4865（https://www.acwing.com/problem/content/4868/）经典栈模拟
5136（https://www.acwing.com/problem/content/description/5139/）经典栈倒序模拟

参考：OI WiKi（xx）
"""
import heapq
import math
from collections import defaultdict, deque
from heapq import heappush, heappop
from itertools import permutations
from math import inf
from typing import List

from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2542(nums1: List[int], nums2: List[int], k: int) -> int:
        # 模板：经典排序后枚举使用堆维护K最大的和，类似LC857
        n = len(nums1)
        ind = list(range(n))
        ind.sort(key=lambda it: -nums2[it])
        ans = 0
        stack = []
        pre = 0
        for i in ind:
            heapq.heappush(stack, nums1[i])
            pre += nums1[i]
            if len(stack) > k:
                pre -= heapq.heappop(stack)
            if len(stack) == k:
                if pre * nums2[i] > ans:
                    ans = pre * nums2[i]
        return ans

    @staticmethod
    def lc_2462(costs: List[int], k: int, candidates: int) -> int:
        # 模板：使用堆进行贪心模拟
        n = len(costs)
        visit = [0] * n
        pre = [[costs[i], i] for i in range(candidates)]
        post = [[costs[i], i] for i in range(n - candidates, n)]
        heapq.heapify(pre)
        heapq.heapify(post)
        pre_ind = candidates
        post_ind = n - candidates - 1
        ans = 0
        for _ in range(k):
            while pre and visit[pre[0][1]]:
                heapq.heappop(pre)
            while len(pre) < candidates and pre_ind < n:
                if not visit[pre_ind]:
                    heapq.heappush(pre, [costs[pre_ind], pre_ind])
                pre_ind += 1

            while post and visit[post[0][1]]:
                heapq.heappop(post)
            while len(post) < candidates and post_ind >= 0:
                if not visit[post_ind]:
                    heapq.heappush(post, [costs[post_ind], post_ind])
                post_ind -= 1

            if pre and post:
                if pre[0][0] <= post[0][0]:
                    c, i = heapq.heappop(pre)
                else:
                    c, i = heapq.heappop(post)
            elif pre:
                c, i = heapq.heappop(pre)
            else:
                c, i = heapq.heappop(post)
            visit[i] = 1
            ans += c
        return ans

    @staticmethod
    def lc_2813(items: List[List[int]], k: int) -> int:
        # 模板：经典思维题排序后枚举，维护长度为k的子序列最大函数值
        items.sort(reverse=True)
        ans = cnt = pre = tp = 0
        dct = defaultdict(list)
        stack = []
        for p, c in items:
            if cnt == k:
                while stack and len(dct[stack[-1]]) == 1:
                    stack.pop()
                if not stack:
                    break
                pre -= dct[stack.pop()].pop()
            else:
                cnt += 1
            pre += p
            dct[c].append(p)
            if len(dct[c]) == 1:
                tp += 1
            stack.append(c)
            if pre + tp * tp > ans:
                ans = pre + tp * tp
        return ans

    @staticmethod
    def lc_1705(apples: List[int], days: List[int]) -> int:
        # 模板：使用堆进行贪心模拟
        n = len(apples)
        ans = i = 0
        stack = []
        while i < n or stack:
            if i < n and apples[i]:
                heappush(stack, [i + days[i] - 1, apples[i]])
            while stack and (stack[0][0] < i or not stack[0][1]):
                heappop(stack)
            if stack:
                stack[0][1] -= 1
                ans += 1
            i += 1
        return ans

    @staticmethod
    def lc_2197(nums: List[int]) -> List[int]:
        # 模板：栈结合 gcd 与 lcm 进行模拟计算
        stack = []
        for num in nums:
            stack.append(num)
            while len(stack) >= 2:
                g = math.gcd(stack[-1], stack[-2])
                if g > 1:
                    stack[-2] = stack[-1] * stack[-2] // g
                    stack.pop()
                else:
                    break
        return stack

    @staticmethod
    def lc_857(quality: List[int], wage: List[int], k: int) -> float:
        # 模板：经典贪心排序枚举，使用堆维护K个最小值的和
        n = len(quality)
        ind = list(range(n))
        ind.sort(key=lambda it: wage[it] / quality[it])
        ans = inf
        pre = 0
        stack = []
        for i in ind:
            heapq.heappush(stack, -quality[i])
            pre += quality[i]
            if len(stack) > k:
                pre += heapq.heappop(stack)
            if len(stack) == k:
                cur = pre * wage[i] / quality[i]
                if cur < ans:
                    ans = cur
        return ans

    @staticmethod
    def cf_1095e(ac=FastIO()):
        # 模板：计算只替换一个字符的情况下括号串是否合法
        n = ac.read_int()
        s = ac.read_str()
        post = [inf] * (n + 1)
        post[-1] = 0
        x = 0
        for i in range(n - 1, -1, -1):
            if s[i] == ")":
                x += 1
            else:
                x -= 1
            if x < 0:
                break
            post[i] = x

        ans = x = 0
        for i in range(n):
            if s[i] == "(" and x >= 1 and x - 1 == post[i + 1]:
                ans += 1
            elif s[i] == ")" and x >= 0 and x + 1 == post[i + 1]:
                ans += 1
            if s[i] == "(":
                x += 1
            else:
                x -= 1
            if x < 0:
                break
        ac.st(ans)
        return

    @staticmethod
    def abc_62d(ac=FastIO()):
        # 模板：经典堆与前后缀结合
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = [-inf] * (3 * n + 1)
        stack = []
        s = 0
        for i in range(3 * n):
            s += nums[i]
            heappush(stack, nums[i])
            if len(stack) > n:
                s -= heappop(stack)
            if i >= n - 1:
                pre[i] = s

        post = [-inf] * (3 * n + 1)
        stack = []
        s = 0
        for i in range(3 * n - 1, -1, -1):
            s -= nums[i]
            heappush(stack, -nums[i])
            if len(stack) > n:
                s -= heappop(stack)
            if 3 * n - i >= n:
                post[i] = s
        ans = max(pre[i] + post[i + 1] for i in range(n - 1, 2 * n + 1))
        ac.st(ans)
        return

    @staticmethod
    def cf_5c(s):
        # 模板：使用栈计算最长连续合法括号子序列以及个数
        stack = [["", -1]]
        ans = cnt = 0
        n = len(s)
        for i in range(n):
            if s[i] == "(":
                stack.append([s[i], i])
            else:
                if stack[-1][0] != "(":
                    stack = [["", i]]
                else:
                    stack.pop()
                    cur = i - stack[-1][1]
                    if cur > ans:
                        ans = cur
                        cnt = 1
                    elif cur == ans:
                        cnt += 1
        if not ans:
            cnt = 1
        return [ans, cnt]

    @staticmethod
    def ac_128(ac=FastIO()):
        # 模板：经典栈与指针模拟
        pre = []
        post = []
        pre_sum = [0]
        pre_ceil = [-inf]
        for _ in range(ac.read_int()):
            lst = ac.read_list_strs()
            if lst[0] == "I":
                pre.append(int(lst[1]))
                pre_sum.append(pre_sum[-1] + int(lst[1]))
                pre_ceil.append(ac.max(pre_ceil[-1], pre_sum[-1]))
            elif lst[0] == "D":
                if pre:
                    pre.pop()
                    pre_sum.pop()
                    pre_ceil.pop()
            elif lst[0] == "L":
                if pre:
                    post.append(pre.pop())
                    pre_sum.pop()
                    pre_ceil.pop()
            elif lst[0] == "R":
                if post:
                    x = post.pop()
                    pre.append(x)
                    pre_sum.append(pre_sum[-1] + x)
                    pre_ceil.append(ac.max(pre_ceil[-1], pre_sum[-1]))
            else:
                ac.st(pre_ceil[int(lst[1])])
        return

    @staticmethod
    def ac_129_1(ac=FastIO()):
        # 模板：经典卡特兰数，栈模拟判定出栈入栈合法性
        n = ac.read_int()
        m = ac.min(5, n)

        pre = list(range(1, n + 1))

        def check(lst):
            lst = deque(lst)
            stack = []
            for num in pre:
                stack.append(num)
                while stack and stack[-1] == lst[0]:
                    stack.pop()
                    lst.popleft()
            return not stack

        cnt = 0
        for item in permutations(list(range(n - m + 1, n + 1)), m):
            cur = list(range(1, n - m + 1)) + list(item)
            if check(cur):
                ac.st("".join(str(x) for x in cur))
                cnt += 1
            if cnt == 20:
                break
        return

    @staticmethod
    def ac_129_2(ac=FastIO()):
        # 模板：使用回溯模拟出栈入栈所有可能的排列

        def dfs(i):
            nonlocal cnt, post, pre
            if cnt >= 20:
                return
            if i == n:
                cnt += 1
                ac.st("".join(str(x) for x in res))
                return

            if pre:
                res.append(pre.pop())
                dfs(i + 1)
                pre.append(res.pop())

            if post:
                pre.append(post.popleft())
                dfs(i)
                post.appendleft(pre.pop())
            return

        n = ac.read_int()
        post = deque(list(range(1, n + 1)))
        res = []
        pre = []
        cnt = 0
        dfs(0)
        return

    @staticmethod
    def ac_129_3(ac=FastIO()):
        # 模板：使用迭代写法替换深搜与回溯
        n = ac.read_int()
        cnt = 0
        stack = [[[], [], 0]]
        while stack and cnt < 20:
            pre, res, ind = stack.pop()
            if len(res) == n:
                cnt += 1
                ac.st("".join(str(x) for x in res))
            else:
                if ind + 1 <= n:
                    stack.append([pre + [ind + 1], res[:], ind + 1])
                if pre:
                    stack.append([pre[:-1], res + [pre[-1]], ind])
        return

    @staticmethod
    def lg_p1974(ac=FastIO()):
        # 模板：贪心队列模拟
        n = ac.read_int()
        stack = deque([1] * n)
        while len(stack) >= 2:
            a, b = stack.popleft(), stack.popleft()
            stack.append(a * b + 1)
        ac.st(stack[0])
        return

    @staticmethod
    def lg_p3719(ac=FastIO()):
        # 模板：栈模拟
        s = ac.read_str()
        stack = []
        for w in s:
            if w != ")":
                stack.append(w)
            else:
                pre = ""
                while stack and stack[-1] != "(":
                    w = stack.pop()
                    pre += w
                stack.pop()
                x = max(len(t) for t in pre.split("|"))
                stack.append("a" * x)

        pre = "".join(stack)
        x = max(len(t) for t in pre.split("|"))
        ac.st(x)
        return

    @staticmethod
    def ac_4865(ac=FastIO()):
        # 模板：经典栈模拟
        m = ac.read_int()
        lst = ac.read_list_strs()
        n = len(lst)
        if n != m * 2 - 1:
            ac.st("Error occurred")
            return
        if m == 1:
            ac.st("int")
            return

        stack = []
        for i in range(n):
            if lst[i] == "int":
                stack.append([[i, i], "int"])
                # 维护每个函数段的左右边界
                while len(stack) >= 3 and [ls[1] for ls in stack[-3:]] == ["pair", "int", "int"]:
                    lst[stack[-1][0][1]] += ">"
                    lst[stack[-2][0][1]] += ","
                    lst[stack[-2][0][0]] = "<" + lst[stack[-2][0][0]]
                    stack[-3][0][1] = stack[-1][0][1]
                    stack[-3][0][0] = stack[-3][0][0]
                    stack[-3][1] = "int"
                    stack.pop()
                    stack.pop()
            else:
                stack.append([[i, i], "pair"])

        if len(stack) > 1:
            ac.st("Error occurred")
            return
        ac.st("".join(lst))
        return

    @staticmethod
    def ac_5136(ac=FastIO()):
        # 模板：经典栈倒序模拟
        s = ac.read_str()
        n = len(s)
        ans = [0] * n
        right = 0
        post = deque()
        for i in range(n - 1, -1, -1):
            if s[i] == "#":
                post.append(i)
            elif s[i] == "(":
                if right:
                    right -= 1
                else:
                    if not post:
                        ac.st(-1)
                        return
                    ans[post[0]] += 1
            else:
                right += 1
            while len(post) >= 2 and ans[post[0]]:
                post.popleft()
        while post and ans[post[0]]:
            post.popleft()
        if post or right:
            ac.st(-1)
            return
        for i in range(n):
            if s[i] == "#":
                ac.st(ans[i])
        return