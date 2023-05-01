import unittest

from algorithm.src.fast_io import FastIO

"""
算法：分治、递归、二叉树、四叉树、十叉树、N叉树、先序、中序、后序遍历，也叫（divide and conquer）
功能：递归进行处理，与迭代是处理相同问题的两种不同方式，迭代效率高于递归
题目：

===================================洛谷===================================
P1911 L 国的战斗之排兵布阵（https://www.luogu.com.cn/problem/P1911）使用四叉树递归计算
P5461 赦免战俘（https://www.luogu.com.cn/problem/P5461）递归计算四叉树左上角
P5551 Chino的树学（https://www.luogu.com.cn/problem/P5551）先序遍历的完全二叉树递归计算
P5626 【AFOI-19】数码排序（https://www.luogu.com.cn/problem/P5626）分治DP，归并排序需要的比较次数最少，但是可能内存占用超过快排
P2907 [USACO08OPEN]Roads Around The Farm S（https://www.luogu.com.cn/problem/P2907）分析复杂度之后采用递归模拟
P7673 [COCI2013-2014#5] OBILAZAK（https://www.luogu.com.cn/problem/P7673）根据中序遍历，递归还原完全二叉树
P1228 地毯填补问题（https://www.luogu.com.cn/problem/P1228）四叉树分治递归

================================CodeForces================================
C. Painting Fence（https://codeforces.com/contest/448/problem/C）贪心递归DP

98. 分形之城（https://www.acwing.com/problem/content/100/）四叉树递归与坐标旋转变换
93. 递归实现组合型枚举（https://www.acwing.com/problem/content/95/）递归与迭代两种方式实现组合数选取
118. 分形（https://www.acwing.com/problem/content/120/）递归生成分形

参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1911(n, x, y):

        # 模板：递归处理四叉树

        def dfs(x1, y1, x2, y2, a, b):
            nonlocal ind
            if x1 == x2 and y1 == y2:
                return

            # 确定是哪一个角被占用了
            flag = find(x1, y1, x2, y2, a, b)
            x0 = x1 + (x2 - x1) // 2
            y0 = y1 + (y2 - y1) // 2

            # 四叉树中心邻居节点
            lst = [[x0, y0], [x0, y0 + 1], [x0 + 1, y0], [x0 + 1, y0 + 1]]
            nex = []
            for i in range(4):
                if i != flag:
                    ans[lst[i][0]][lst[i][1]] = ind
                    nex.append(lst[i])
                else:
                    nex.append([a, b])
            ind += 1
            # 四叉树递归坐标
            dfs(x1, y1, x0, y0, nex[0][0], nex[0][1])
            dfs(x1, y0 + 1, x0, y2, nex[1][0], nex[1][1])
            dfs(x0 + 1, y1, x2, y0, nex[2][0], nex[2][1])
            dfs(x0 + 1, y0 + 1, x2, y2, nex[3][0], nex[3][1])
            return

        def find(x1, y1, x2, y2, a, b):
            x0 = x1 + (x2 - x1) // 2
            y0 = y1 + (y2 - y1) // 2
            if x1 <= a <= x0 and y1 <= b <= y0:
                return 0
            if x1 <= a <= x0 and y0 + 1 <= b <= y2:
                return 1
            if x0 + 1 <= a <= x2 and y1 <= b <= y0:
                return 2
            return 3

        x -= 1
        y -= 1
        m = 1 << n
        ans = [[0] * m for _ in range(m)]
        ind = 1
        # 递归生成
        dfs(0, 0, m - 1, m - 1, x, y)

        # 哈希化处理
        dct = dict()
        dct[0] = 0
        for i in range(m):
            for j in range(m):
                x = ans[i][j]
                if x not in dct:
                    dct[x] = len(dct)
        return [[dct[i] for i in a] for a in ans]

    @staticmethod
    def cf_448c(ac=FastIO()):
        # 模板：贪心递归DP
        n = ac.read_int()
        nums = ac.read_list_ints()

        @ac.bootstrap
        def dfs(arr):
            m = len(arr)
            low = min(arr)
            cur = [num-low for num in arr]
            ans = low
            i = 0
            while i < m:
                if cur[i] == 0:
                    i += 1
                    continue
                j = i
                while j < m and cur[j] > 0:
                    j += 1
                ans += yield dfs(cur[i: j])
                i = j
            yield ac.min(ans, m)

        ac.st(dfs(nums))
        return

    @staticmethod
    def ac_98(ac=FastIO()):

        # 模板：四叉树递归与坐标旋转变换
        for _ in range(ac.read_int()):
            n, a, b = ac.read_ints()
            a -= 1
            b -= 1

            def check(nn, mm):
                # 递归改成迭代写法极大提升速度
                stack = [[nn, mm]]
                x = y = -1
                while stack:
                    if stack[-1][0] == 0:
                        x = y = 0
                        stack.pop()
                        continue
                    else:
                        nn, mm = stack[-1]
                        cc = 2 ** (2 * nn - 2)
                        if x != -1:
                            stack.pop()
                            z = mm // cc
                            length = 2 ** (nn - 1)
                            if z == 0:
                                x, y = y, x
                            elif z == 1:
                                x, y = x, y + length
                            elif z == 2:
                                x, y = x + length, y + length
                            else:
                                x, y = 2 * length - y - 1, length - x - 1
                        else:
                            stack.append([nn-1, mm % cc])
                return x, y

            x1, y1 = check(n, a)
            x2, y2 = check(n, b)
            ans = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 * 10
            ans = int(ans) + int(ans - int(ans) >= 0.5)
            ac.st(ans)
        return

    @staticmethod
    def ac_93_1(ac=FastIO()):
        n, m = ac.read_ints()
        # 模板：递归实现选取

        def dfs(i):
            if len(pre) == m:
                ac.lst(pre)
                return
            if i == n:
                return

            dfs(i + 1)
            pre.append(i + 1)
            dfs(i + 1)
            pre.pop()
            return

        pre = []
        dfs(0)
        return

    @staticmethod
    def ac_93_2(ac=FastIO()):
        n, m = ac.read_ints()

        # 模板：迭代实现选取
        pre = []
        stack = [[0, 0]]
        while stack:
            i, state = stack.pop()
            if i >= 0:
                stack.append([~i, state])
                if len(pre) == m:
                    ac.lst(pre)
                    continue
                if i == n:
                    continue
                stack.append([i + 1, 0])
                pre.append(i + 1)
                stack.append([i + 1, 1])
            else:
                if state:
                    pre.pop()
        return

    @staticmethod
    def ac_118(ac=FastIO()):
        # 模板：使用迭代方式进行递归计算
        dp = []
        for i in range(1, 8):
            n = 3 ** (i - 1)
            ans = [[" "] * n for _ in range(n)]
            stack = [[0, 0, n - 1, n - 1]]
            while stack:
                x1, y1, x2, y2 = stack.pop()
                if x1 == x2 and y1 == y2:
                    ans[x1][y1] = "X"
                    continue
                m = x2 - x1 + 1
                b = m // 3
                col = {0: 0, 2: 2, 4: 1, 6: 0, 8: 2}
                for x in [0, 2, 4, 6, 8]:
                    start = [x1 + b * (x // 3), y1 + b * col[x]]
                    end = [x1 + b * (x // 3) + b - 1, y1 + b * col[x] + b - 1]
                    stack.append([start[0], start[1], end[0], end[1]])
            dp.append(["".join(a) for a in ans])

        while True:
            n = ac.read_int()
            if n == -1:
                break
            for a in dp[n-1]:
                ac.st(a)
            ac.st("-")

        return


class TestGeneral(unittest.TestCase):

    def test_rescursion(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
