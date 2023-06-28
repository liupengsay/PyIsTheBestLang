import bisect
import heapq
import random
import unittest
from collections import defaultdict
from typing import List

from algorithm.src.data_structure.sparse_table import SparseTable1
from algorithm.src.fast_io import FastIO

"""
算法：单调栈
功能：用来计算数组前后的更大值更小值信息
题目：

===================================力扣===================================
85. 最大矩形（https://leetcode.cn/problems/maximal-rectangle/）枚举矩形下边界，使用单调栈计算最大矩形面积 
321. 拼接最大数（https://leetcode.cn/problems/create-maximum-number/）经典枚举加单调栈
2334. 元素值大于变化阈值的子数组（https://leetcode.cn/problems/subarray-with-elements-greater-than-varying-threshold/）排序后枚举最小值左右两边的影响范围
2262. 字符串的总引力（https://leetcode.cn/problems/total-appeal-of-a-string/）计算下一个或者上一个不同字符的位置
2355. 你能拿走的最大图书数量（https://leetcode.cn/problems/maximum-number-of-books-you-can-take/）单调栈加DP
255. 验证前序遍历序列二叉搜索树（https://leetcode.cn/problems/verify-preorder-sequence-in-binary-search-tree/）单调栈经典使用，判断数组是否为二叉搜索树的前序遍历，同样地可验证后序遍历
654. 最大二叉树（https://leetcode.cn/problems/maximum-binary-tree/）经典单调栈应用题

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
P1578 奶牛浴场（https://www.luogu.com.cn/problem/P1578）使用单调栈离散化枚举障碍点的最大面积矩形
P3467 [POI2008]PLA-Postering（https://www.luogu.com.cn/problem/P3467）贪心单调栈
P1191 矩形（https://www.luogu.com.cn/problem/P1191）经典单调栈求矩形个数
P1323 删数问题（https://www.luogu.com.cn/problem/P1323）二叉堆与单调栈，计算最大字典序数字
P2422 良好的感觉（https://www.luogu.com.cn/problem/P2422）单调栈与前缀和
P3467 [POI2008]PLA-Postering（https://www.luogu.com.cn/problem/P3467）看不懂的单调栈
P6404 [COCI2014-2015#2] BOB（https://www.luogu.com.cn/problem/P6404）经典单调栈计算具有相同数字的子矩形个数
P6503 [COCI2010-2011#3] DIFERENCIJA（https://www.luogu.com.cn/problem/P6503）经典单调栈连续子序列的最大值最小值贡献计数
P6510 奶牛排队（https://www.luogu.com.cn/problem/P6510）单调栈稀疏表加哈希二分
P6801 [CEOI2020] 花式围栏（https://www.luogu.com.cn/problem/P6801）经典单调栈计算矩形个数
P8094 [USACO22JAN] Cow Frisbee S（https://www.luogu.com.cn/problem/P8094）单调栈典型应用前一个更大与后一个更大

================================CodeForces================================
E. Explosions?（https://codeforces.com/problemset/problem/1795/E）单调栈贪心计数枚举，前后缀DP转移


================================AcWing====================================
131. 直方图中最大的矩形（https://www.acwing.com/problem/content/133/）单调栈求最大矩形
152. 城市游戏（https://www.acwing.com/problem/content/description/154/）单调栈求最大矩形

参考：OI WiKi（xx）
"""


class QuickMonotonicStack:
    def __init__(self):
        return

    @staticmethod
    def pipline(nums):
        # 经典单调栈灵活求解
        n = len(nums)
        post = [-1] * n  # [n-1] * n
        pre = [-1] * n  # [0] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                post[stack.pop()] = i  # i - 1
            if stack:
                pre[i] = stack[-1]  # stack[-1] + 1
            stack.append(i)
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


class Solution:
    def __init__(self):
        return

    @staticmethod
    def ac_131(ac=FastIO()):
        # 模板：单调栈计算最大矩形
        while True:
            lst = ac.read_list_ints()
            if lst[0] == 0:
                break
            n = lst.pop(0)
            post = [n-1]*n
            pre = [0]*n
            stack = []
            for i in range(n):
                while stack and lst[stack[-1]] > lst[i]:
                    post[stack.pop()] = i-1
                if stack:
                    pre[i] = stack[-1] + 1
                stack.append(i)
            ans = max(lst[i]*(post[i]-pre[i]+1) for i in range(n))
            ac.st(ans)
        return

    @staticmethod
    def lg_p1191(ac=FastIO()):
        # 模板：枚举下边界使用单调栈计算矩形个数
        n = ac.read_int()
        pre = [0]*n
        ans = 0
        for _ in range(n):
            s = ac.read_str()
            right = [n-1]*n
            left = [0]*n
            stack = []
            for j in range(n):
                if s[j] == "W":
                    pre[j] += 1
                else:
                    pre[j] = 0
                while stack and pre[stack[-1]] > pre[j]:
                    right[stack.pop()] = j-1
                if stack:
                    left[j] = stack[-1] + 1
                stack.append(j)
            ans += sum(pre[j]*(right[j]-j+1)*(j-left[j]+1) for j in range(n))
        ac.st(ans)
        return

    @staticmethod
    def lg_p1323(ac=FastIO()):
        # 模板：二叉堆与单调栈，计算最大字典序数字
        k, m = ac.read_ints()
        dct = set()
        ans = []
        stack = [1]
        while len(ans) < k:
            num = heapq.heappop(stack)
            if num in dct:
                continue
            ans.append(num)
            dct.add(num)
            heapq.heappush(stack, 2*num+1)
            heapq.heappush(stack, 4 * num + 5)

        res = "".join(str(x) for x in ans)
        ac.st(res)
        rem = m
        stack = []
        for w in res:
            while stack and rem and w > stack[-1]:
                stack.pop()
                rem -= 1
            stack.append(w)
        stack = stack[rem:]
        ac.st(int("".join(stack)))
        return

    @staticmethod
    def lg_p2422(ac=FastIO()):
        # 模板：单调栈与前缀和
        n = ac.read_int()
        nums = ac.read_list_ints()
        lst = ac.accumulate(nums)
        post = [n - 1] * n
        pre = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                post[stack.pop()] = i - 1
            if stack:
                pre[i] = stack[-1] + 1
            stack.append(i)
        ans = max(nums[i] * (lst[post[i] + 1] - lst[pre[i]]) for i in range(n))
        ac.st(ans)
        return


class Rectangle:
    def __init__(self):
        return

    @staticmethod
    def compute_area(pre):
        # 模板：使用单调栈根据高度计算最大矩形面积

        m = len(pre)
        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and pre[stack[-1]] > pre[i]:
                right[stack.pop()] = i - 1
            if stack:  # 这里可以同时求得数组前后的下一个大于等于值
                left[i] = stack[-1] + 1  # 这里将相同的值视为右边的更大且并不会影响计算
            stack.append(i)

        ans = 0
        for i in range(m):
            cur = pre[i] * (right[i] - left[i] + 1)
            ans = ans if ans > cur else cur
        return ans

    @staticmethod
    def compute_number(pre):
        # 模板：使用单调栈根据高度计算矩形个数

        n = len(pre)
        right = [n - 1] * n
        left = [0] * n
        stack = []
        for j in range(n):
            while stack and pre[stack[-1]] > pre[j]:
                right[stack.pop()] = j - 1
            if stack:  # 这个单调栈过程和上述求面积的一样
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
    def lg_p3467(ac=FastIO()):
        # 模板：使用单调栈进行计算
        n = ac.read_int()
        nums = [ac.read_list_ints()[1] for _ in range(n)]
        stack = []
        ans = 0
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                j = stack.pop()
                if nums[j] == nums[i]:  # 同样高度的连续区域可以一次覆盖
                    ans += 1
            stack.append(i)
        ac.st(n - ans)
        return

    @staticmethod
    def lg_p1598(ac=FastIO()):

        # 模板：使用单调栈离散化枚举障碍点的最大面积矩形
        def compute_area_obstacle(lst):
            nonlocal ans
            # 模板：使用单调栈根据高度计算最大矩形面积
            m = len(height)
            left = [0] * m
            right = [m - 1] * m
            stack = []
            for i in range(m):
                while stack and height[stack[-1]] > height[i]:
                    right[stack.pop()] = i  # 注意这里不减 1 了是边界
                if stack:  # 这里可以同时求得数组前后的下一个大于等于值
                    left[i] = stack[-1]  # 这里将相同的值视为右边的更大且并不会影响计算，注意这里不加 1 了是边界
                stack.append(i)

            for i in range(m):
                cur = height[i] * (lst[right[i]] - lst[left[i]])
                ans = ans if ans > cur else cur
            return ans

        length, n = ac.read_ints()
        q = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(q)]
        node_row = defaultdict(list)
        node_col = defaultdict(list)
        for x, y in nums:
            node_row[y].append(x)
            node_col[x].append(y)

        # 枚举矩形上下两行边界
        y_axis = sorted([y for _, y in nums]+[0, n], reverse=True)
        ans = 0
        col = defaultdict(lambda: n)
        x_axis = sorted([x for x, _ in nums]+[0, length])
        for y in y_axis:
            height = [col[x] - y for x in x_axis]
            compute_area_obstacle(x_axis)
            for x in node_row[y]:
                col[x] = y

        # 枚举矩形左右两列边界
        x_axis.reverse()
        y_axis.reverse()
        row = defaultdict(lambda: length)
        for x in x_axis:
            height = [row[y] - x for y in y_axis]
            compute_area_obstacle(y_axis)
            for y in node_col[x]:
                row[y] = x
        ac.st(ans)
        return

    @staticmethod
    def lc_255(preorder: List[int]) -> bool:
        # 模板：使用单调栈判断是否为前序序列

        pre_max = float("-inf")
        n = len(preorder)
        stack = []
        for i in range(n):
            if preorder[i] < pre_max:
                return False
            while stack and preorder[stack[-1]] < preorder[i]:
                cur = preorder[stack.pop()]
                pre_max = pre_max if pre_max > cur else cur
            stack.append(i)
        return True

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

    @staticmethod
    def lg_p6404(ac=FastIO()):
        # 模板：经典单调栈计算具有相同数字的子矩形个数
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        ans = 0
        rt = Rectangle()
        pre = [[0, 0] for _ in range(n)]
        # 枚举子矩形的下边界
        for i in range(m):
            for j in range(n):
                if pre[j][0] == grid[i][j]:
                    pre[j][1] += 1
                else:
                    pre[j] = [grid[i][j], 1]
            # 按照相同数字分段计数
            lst = [pre[0][1]]
            num = pre[0][0]
            for x, c in pre[1:]:
                if x == num:
                    lst.append(c)
                else:
                    ans += rt.compute_number(lst)
                    lst = [c]
                    num = x
            ans += rt.compute_number(lst)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6503(ac=FastIO()):
        # 模板：经典单调栈连续子序列的最大值最小值贡献计数
        m = ac.read_int()
        nums = [ac.read_int() for _ in range(m)]
        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and nums[stack[-1]] < nums[i]:
                right[stack.pop()] = i - 1
            if stack:  # 这里可以同时求得数组前后的下一个大于等于值
                left[i] = stack[-1] + 1  # 这里将相同的值视为右边的更大且并不会影响计算
            stack.append(i)
        ans = sum((right[i]-i+1)*nums[i]*(i-left[i]+1) for i in range(m))

        left = [0] * m
        right = [m - 1] * m
        stack = []
        for i in range(m):
            while stack and nums[stack[-1]] > nums[i]:
                right[stack.pop()] = i - 1
            if stack:  # 这里可以同时求得数组前后的下一个大于等于值
                left[i] = stack[-1] + 1  # 这里将相同的值视为右边的更大且并不会影响计算
            stack.append(i)
        ans -= sum((right[i]-i+1)*nums[i]*(i-left[i]+1) for i in range(m))
        ac.st(ans)
        return

    @staticmethod
    def lg_p6510(ac=FastIO()):
        # 模板：单调栈稀疏表加哈希二分
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        post = [n-1]*n
        stack = []
        dct = defaultdict(list)
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                post[stack.pop()] = i-1
            stack.append(i)
            dct[nums[i]].append(i)
        st = SparseTable1(nums)
        ans = 0
        for i in range(n):
            x = st.query(i+1, post[i]+1)
            if x == nums[i]:
                continue
            j = bisect.bisect_left(dct[x], i)
            ans = ac.max(ans, dct[x][j]-i+1)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6801(ac=FastIO()):
        # 模板：经典单调栈计算矩形个数

        def compute(x, y):
            return x * (x + 1) * y * (y + 1) // 4

        ans = 0
        mod = 10 ** 9 + 7
        n = ac.read_int()
        h = ac.read_list_ints()
        w = ac.read_list_ints()
        # 削峰维持单调递增
        stack = []
        for i in range(n):
            ww, hh = w[i], h[i]
            while stack and stack[-1][1] >= hh:
                www, hhh = stack.pop()
                if stack and stack[-1][1] >= hh:
                    max_h = stack[-1][1]
                    ans += compute(www, hhh) - compute(www, max_h)
                    ans %= mod
                    stack[-1][0] += www
                else:
                    ww += www
                    ans += compute(www, hhh) - compute(www, hh)
                    ans %= mod
            stack.append([ww, hh])
        # 反向计算剩余
        ww, hh = stack.pop()
        while stack:
            www, hhh = stack.pop()
            ans += compute(ww, hh) - compute(ww, hhh)
            ans %= mod
            ww += www
            hh = hhh
        ans += compute(ww, hh)
        ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def lg_p8094(ac=FastIO()):
        # 模板：经典单调栈应用
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = 0
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                ans += i - stack.pop() + 1   # 当前作为较大值
            if stack:
                ans += i - stack[-1] + 1  # 当前作为较小值
            stack.append(i)
        ac.st(ans)
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
