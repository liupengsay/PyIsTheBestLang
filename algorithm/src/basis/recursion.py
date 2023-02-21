import unittest

"""
算法：分治、递归、二叉树、四叉树、十叉树、N叉树、先序、中序、后序遍历
功能：递归进行处理，与迭代是处理相同问题的两种不同方式
题目：

===================================洛谷===================================
P1911 L 国的战斗之排兵布阵（https://www.luogu.com.cn/problem/P1911）使用四叉树递归计算
P5461 赦免战俘（https://www.luogu.com.cn/problem/P5461）递归计算四叉树左上角
P5551 Chino的树学（https://www.luogu.com.cn/problem/P5551）先序遍历的完全二叉树递归计算
P5626 【AFOI-19】数码排序（https://www.luogu.com.cn/problem/P5626）分治DP，归并排序需要的比较次数最少，但是可能内存占用超过快排
P2907 [USACO08OPEN]Roads Around The Farm S（https://www.luogu.com.cn/problem/P2907）分析复杂度之后采用递归模拟
P7673 [COCI2013-2014#5] OBILAZAK（https://www.luogu.com.cn/problem/P7673）根据中序遍历，递归还原完全二叉树


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


class TestGeneral(unittest.TestCase):

    def test_rescursion(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
