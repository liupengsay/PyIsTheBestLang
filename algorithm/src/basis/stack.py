import math
import unittest
from collections import deque
from itertools import permutations
from typing import List

from algorithm.src.fast_io import FastIO, inf

"""

算法：栈、RBS（合法括号子序列）
功能：模拟题中常见，如括号之类的，后进先出，升级版应用有单调栈、最大栈和最小栈
题目：

===================================力扣===================================
2197. 替换数组中的非互质数（https://leetcode.cn/problems/replace-non-coprime-numbers-in-array/）结合数学使用栈进行模拟
394. 字符串解码（https://leetcode.cn/problems/decode-string/）经典解码带括号成倍的字符和数字
1096. 花括号展开 II（https://leetcode.cn/problems/brace-expansion-ii/）使用栈进行字符解码

===================================洛谷===================================
P1944 最长括号匹配（https://www.luogu.com.cn/problem/P1944）最长连续合法括号字串长度
P2201 数列编辑器（https://www.luogu.com.cn/problem/P2201）双栈模拟指针移动同时记录前缀和与前序最大前缀和
P4387 【深基15.习9】验证栈序列（https://www.luogu.com.cn/problem/P4387）模拟入栈出栈队列判断是否可行
P7674 [COCI2013-2014#5] EKSPLOZIJA（https://www.luogu.com.cn/problem/P7674）使用栈模仿消除
P3719 [AHOI2017初中组]rexp（https://www.luogu.com.cn/problem/P3719）字符串运算展开
P1974 基因聚合（https://www.luogu.com.cn/problem/P1974）贪心队列模拟
P3551 [POI2013]USU-Take-out（https://www.luogu.com.cn/problem/P3551）栈与计数指针
P3719 [AHOI2017初中组]rexp（https://www.luogu.com.cn/problem/P3719）栈模拟

================================CodeForces================================
C. Longest Regular Bracket Sequence（https://codeforces.com/problemset/problem/5/C）最长连续合法括号子序列以及个数
E. Almost Regular Bracket Sequence（https://codeforces.com/problemset/problem/1095/E）计算改变一个括号后是的字符串合法的位置数


================================AcWing===================================
128. 编辑器（https://www.acwing.com/problem/content/130/）堆栈模拟
129. 火车进栈（https://www.acwing.com/problem/content/131/）经典卡特兰数，栈模拟判定出栈入栈合法性
132. 小组队列（https://www.acwing.com/problem/content/134/）双端队列依次出队入队

参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_2197(nums: List[int]) -> List[int]:
        # 模板：栈结合 gcd 与 lcm 进行模拟计算
        stack = list()
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


class MaxStack:
    # 模板：经典最大栈
    def __init__(self):
        return

    def gen_result(self):
        return


class MinStack:
    # 模板：经典最小栈
    def __init__(self):
        return

    def gen_result(self):
        return

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
                pre_sum.append(pre_sum[-1]+int(lst[1]))
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

        pre = list(range(1, n+1))

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
        for item in permutations(list(range(n-m+1, n+1)), m):
            cur = list(range(1, n-m+1))+list(item)
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
                dfs(i+1)
                pre.append(res.pop())

            if post:
                pre.append(post.popleft())
                dfs(i)
                post.appendleft(pre.pop())
            return

        n = ac.read_int()
        post = deque(list(range(1, n+1)))
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
                    stack.append([pre+[ind+1], res[:], ind+1])
                if pre:
                    stack.append([pre[:-1], res+[pre[-1]], ind])
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


class TestGeneral(unittest.TestCase):

    def test_stack(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
