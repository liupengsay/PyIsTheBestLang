import unittest

"""

算法：栈
功能：模拟题中常见，如括号之类的，后进先出，升级版应用有单调栈、最大栈和最小栈
题目：

===================================力扣===================================
2197. 替换数组中的非互质数（https://leetcode.cn/problems/replace-non-coprime-numbers-in-array/）结合数学使用栈进行模拟
394. 字符串解码（https://leetcode.cn/problems/decode-string/）经典解码带括号成倍的字符和数字

===================================洛谷===================================
P1944 最长括号匹配（https://www.luogu.com.cn/problem/P1944）最长连续合法括号字串长度
P2201 数列编辑器（https://www.luogu.com.cn/problem/P2201）双栈模拟指针移动同时记录前缀和与前序最大前缀和
P4387 【深基15.习9】验证栈序列（https://www.luogu.com.cn/problem/P4387）模拟入栈出栈队列判断是否可行
P7674 [COCI2013-2014#5] EKSPLOZIJA（https://www.luogu.com.cn/problem/P7674）使用栈模仿消除

================================CodeForces================================
C. Longest Regular Bracket Sequence（https://codeforces.com/problemset/problem/5/C）最长连续合法括号子序列以及个数


参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
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


class Stack:
    def __init__(self):
        return

    def gen_result(self):
        return


class MaxStack:
    def __init__(self):
        return

    def gen_result(self):
        return


class MinStack:
    def __init__(self):
        return

    def gen_result(self):
        return


class TestGeneral(unittest.TestCase):

    def test_stack(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
