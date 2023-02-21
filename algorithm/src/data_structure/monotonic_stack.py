import random
import unittest
from typing import List

from algorithm.src.fast_io import FastIO

"""
算法：单调栈
功能：用来计算数组前后的更大值更小值信息
题目：

===================================力扣===================================
85. 最大矩形（https://leetcode.cn/problems/maximal-rectangle/）枚举矩形下边界，使用单调栈计算最大矩形面积 
2334. 元素值大于变化阈值的子数组（https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/）排序后枚举最小值左右两边的影响范围
2262. 字符串的总引力（https://leetcode.cn/problems/total-appeal-of-a-string/）计算下一个或者上一个不同字符的位置

===================================洛谷===================================
P1950 长方形（https://www.luogu.com.cn/problem/P1950）通过枚举下边界，结合单调栈计算矩形个数
P1901 发射站（https://www.luogu.com.cn/problem/P1901）由不相同的数组成的数组求其前后的更大值
P2866 [USACO06NOV]Bad Hair Day S（https://www.luogu.com.cn/problem/P2866）单调栈
P2947 [USACO09MAR]Look Up S（https://www.luogu.com.cn/problem/P2947）单调栈裸题
P4147 玉蟾宫（https://www.luogu.com.cn/problem/P4147）枚举矩形的下边界，使用单调栈计算最大矩形面积
P5788 【模板】单调栈（https://www.luogu.com.cn/problem/P5788）单调栈模板题
P7314 [COCI2018-2019#3] Pismo（https://www.luogu.com.cn/problem/P7314）枚举当前最小值，使用单调栈确定前后第一个比它大的值
P7399 [COCI2020-2021#5] Po（https://www.luogu.com.cn/problem/P7399）单调栈变形题目，贪心进行赋值，区间操作达成目标数组
P7410 [USACO21FEB] Just Green Enough S（https://www.luogu.com.cn/problem/P7410）通过容斥原理与单调栈计算01矩阵个数
P7762 [COCI2016-2017#5] Unija（https://www.luogu.com.cn/problem/P7762）类似单调栈的思想，按照宽度进行贪心排序，计算每个高度的面积贡献

================================CodeForces================================
E. Explosions?（https://codeforces.com/problemset/problem/1795/E）单调栈贪心计数枚举，前后缀DP转移



参考：OI WiKi（xx）
"""

class Rectangle:
    def __init__(self):
        return

    @staticmethod
    def compute_area(pre):
        # 根据高度计算最大矩形面积
        m = len(pre)
        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and pre[stack[-1]] > pre[i]:
                right[stack.pop()] = i - 1
            if stack:
                left[i] = stack[-1] + 1
            stack.append(i)

        ans = 0
        for i in range(m):
            cur = pre[i] * (right[i] - left[i] + 1)
            ans = ans if ans > cur else cur
        return ans

    @staticmethod
    def compute_number(pre):
        n = len(pre)
        # 根据高度计算矩形个数
        right = [n - 1] * n
        left = [0] * n
        stack = []
        for j in range(n):
            while stack and pre[stack[-1]] > pre[j]:
                right[stack.pop()] = j - 1
            if stack:
                left[j] = stack[-1] + 1
            stack.append(j)

        ans = 0
        for j in range(n):
            ans += (right[j] - j + 1) * (j - left[j] + 1) * pre[j]
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_85(matrix: List[List[str]]) -> int:
        # 模板：单调栈计算最大矩形面积
        m, n = len(matrix), len(matrix[0])
        pre = [0] * n
        ans = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    pre[j] += 1
                else:
                    pre[j] = 0
            ans = max(ans, Rectangle().compute_area(pre))
        return ans

    @staticmethod
    def lg_p4147(ac=FastIO()):
        # 模板：单调栈计算最大矩形面积
        n, m = ac.read_ints()
        pre = [0] * m
        ans = 0
        for _ in range(n):
            lst = ac.read_list_strs()
            for i in range(m):
                if lst[i] == "F":
                    pre[i] += 1
                else:
                    pre[i] = 0
            ans = ac.max(ans, Rectangle().compute_area(pre))
        ac.st(3 * ans)
        return

    @staticmethod
    def lg_p1950(ac=FastIO()):
        # 模板：单调栈计算矩形个数
        m, n = ac.read_ints()
        ans = 0
        pre = [0] * n
        for _ in range(m):
            s = ac.read_str()
            for j in range(n):
                if s[j] == ".":
                    pre[j] += 1
                else:
                    pre[j] = 0
            ans += Rectangle().compute_number(pre)
        ac.st(ans)
        return


class MonotonicStack:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)

        # 视情况可给不存在前序相关最值的值赋 i 或者 0
        self.pre_bigger = [-1] * self.n  # 上一个更大值
        self.pre_bigger_equal = [-1] * self.n  # 上一个大于等于值
        self.pre_smaller = [-1] * self.n  # 上一个更小值
        self.pre_smaller_equal = [-1] * self.n  # 上一个小于等于值

        # 视情况可给不存在前序相关最值的值赋 i 或者 n-1
        self.post_bigger = [-1] * self.n  # 下一个更大值
        self.post_bigger_equal = [-1] * self.n  # 下一个大于等于值
        self.post_smaller = [-1] * self.n  # 下一个更小值
        self.post_smaller_equal = [-1] * self.n   # 下一个小于等于值

        self.gen_result()
        return

    def gen_result(self):

        # 从前往后遍历
        stack = []
        for i in range(self.n):
            while stack and self.nums[i] >= self.nums[stack[-1]]:
                self.post_bigger_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.pre_bigger[i] = stack[-1]  # 有时也用 stack[-1]+1 做为边界
            stack.append(i)

        stack = []
        for i in range(self.n):
            while stack and self.nums[i] <= self.nums[stack[-1]]:
                self.post_smaller_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.pre_smaller[i] = stack[-1]  # 有时也用 stack[-1]+1 做为边界
            stack.append(i)

        # 从后往前遍历
        stack = []
        for i in range(self.n - 1, -1, -1):
            while stack and self.nums[i] >= self.nums[stack[-1]]:
                self.pre_bigger_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.post_bigger[i] = stack[-1]  # 有时也用 stack[-1]-1 做为边界
            stack.append(i)

        stack = []
        for i in range(self.n - 1, -1, -1):
            while stack and self.nums[i] <= self.nums[stack[-1]]:
                self.pre_smaller_equal[stack.pop()] = i  # 有时也用 i-1 作为边界
            if stack:
                self.post_smaller[i] = stack[-1]  # 有时也用 stack[-1]-1 做为边界
            stack.append(i)

        return


class TestGeneral(unittest.TestCase):

    def test_monotonic_stack(self):
        n = 1000
        nums = [random.randint(0, n) for _ in range(n)]
        ms = MonotonicStack(nums)
        for i in range(n):

            # 上一个最值
            pre_bigger = pre_bigger_equal = pre_smaller = pre_smaller_equal = -1
            for j in range(i - 1, -1, -1):
                if nums[j] > nums[i]:
                    pre_bigger = j
                    break
            for j in range(i - 1, -1, -1):
                if nums[j] >= nums[i]:
                    pre_bigger_equal = j
                    break
            for j in range(i - 1, -1, -1):
                if nums[j] < nums[i]:
                    pre_smaller = j
                    break
            for j in range(i - 1, -1, -1):
                if nums[j] <= nums[i]:
                    pre_smaller_equal = j
                    break
            assert pre_bigger == ms.pre_bigger[i]
            assert pre_bigger_equal == ms.pre_bigger_equal[i]
            assert pre_smaller == ms.pre_smaller[i]
            assert pre_smaller_equal == ms.pre_smaller_equal[i]

            # 下一个最值
            post_bigger = post_bigger_equal = post_smaller = post_smaller_equal = - 1
            for j in range(i + 1, n):
                if nums[j] > nums[i]:
                    post_bigger = j
                    break
            for j in range(i + 1, n):
                if nums[j] >= nums[i]:
                    post_bigger_equal = j
                    break
            for j in range(i + 1, n):
                if nums[j] < nums[i]:
                    post_smaller = j
                    break
            for j in range(i + 1, n):
                if nums[j] <= nums[i]:
                    post_smaller_equal = j
                    break
            assert post_bigger == ms.post_bigger[i]
            assert post_bigger_equal == ms.post_bigger_equal[i]
            assert post_smaller == ms.post_smaller[i]
            assert post_smaller_equal == ms.post_smaller_equal[i]

        return


if __name__ == '__main__':
    unittest.main()
