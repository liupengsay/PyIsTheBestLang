"""

"""
"""
算法：深度优先搜索
功能：常与回溯枚举结合使用，比较经典的还有DFS序
题目：

P2383 狗哥玩木棒（https://www.luogu.com.cn/problem/P2383）暴力搜索木棍拼接组成正方形
473. 火柴拼正方形（https://leetcode.cn/problems/matchsticks-to-square/）暴力搜索木棍拼接组成正方形
P1120 小木棍（https://www.luogu.com.cn/problem/P1120）把数组分成和相等的子数组
P1692 部落卫队（https://www.luogu.com.cn/problem/P1692）暴力搜索枚举字典序最大可行的连通块

P1612 [yLOI2018] 树上的链（https://www.luogu.com.cn/problem/P1612）使用dfs记录路径的前缀和并使用二分确定最长链条
P1475 [USACO2.3]控制公司 Controlling Companies（https://www.luogu.com.cn/problem/P1475）深搜确定可以控制的公司对

P2080 增进感情（https://www.luogu.com.cn/problem/P2080）深搜回溯与剪枝
301. 删除无效的括号（https://leetcode.cn/problems/remove-invalid-parentheses/）深搜回溯与剪枝
P2090 数字对（https://www.luogu.com.cn/problem/P2090）深搜贪心回溯剪枝与辗转相减法
P2420 让我们异或吧（https://www.luogu.com.cn/problem/P2420）脑筋急转弯使用深搜确定到根路径的异或结果以及异或特性获得任意两点之间最短路的异或结果
P1473 [USACO2.3]零的数列 Zero Sum（https://www.luogu.com.cn/problem/P1473）深搜枚举符号数
P1461 [USACO2.1]海明码 Hamming Codes（https://www.luogu.com.cn/problem/P1461）汉明距离计算与深搜回溯枚举
P1394 山上的国度（https://www.luogu.com.cn/problem/P1394）深搜进行可达性确认

P1180 驾车旅游（https://www.luogu.com.cn/problem/P1180）深搜进行模拟
P1118 [USACO06FEB]Backward Digit Sums G/S（https://www.luogu.com.cn/problem/P1118）使用单位矩阵模拟计算杨辉三角的系数，再进行暴搜寻找最小字典序结果
P3252 [JLOI2012]树（https://www.luogu.com.cn/problem/P3252）深搜回溯加前缀和哈希
P4913 【深基16.例3】二叉树深度（https://www.luogu.com.cn/problem/P4913）深搜确定深度
P5118 [USACO18DEC]Back and Forth B（https://www.luogu.com.cn/problem/P5118）深搜回溯与哈希记录进行模拟

P5197 [USACO19JAN]Grass Planting S（https://www.luogu.com.cn/problem/P5197）树形DP模拟与染色法，利用父亲与自己的染色确定儿子们的染色
P5198 [USACO19JAN]Icy Perimeter S（https://www.luogu.com.cn/problem/P5198）经典计算连通块的周长与面积
P5318 【深基18.例3】查找文献（https://www.luogu.com.cn/problem/P5318）经典广搜拓扑排序与深搜序生成与获取


参考：OI WiKi（xx）
"""

import unittest

from typing import List


class DFS:
    def __init__(self):
        return

    @staticmethod
    def makesquare(matchsticks: List[int]) -> bool:
        # 模板: 深搜将数组分组组成正方形
        n, s = len(matchsticks), sum(matchsticks)
        if s % 4 or max(matchsticks) > s // 4:
            return False

        def dfs(i):
            nonlocal ans
            if ans:
                return
            if i == n:
                if len(pre) == 4:
                    ans = True
                return
            if len(pre) > 4:
                return
            for j in range(len(pre)):
                if pre[j] + matchsticks[i] <= m:
                    pre[j] += matchsticks[i]
                    dfs(i + 1)
                    pre[j] -= matchsticks[i]
            pre.append(matchsticks[i])
            dfs(i + 1)
            pre.pop()
            return

        matchsticks.sort(reverse=True)
        m = s // 4
        ans = False
        pre = []
        dfs(0)
        return ans

    @staticmethod
    def gen_node_order(dct):
        # 生成深搜序即 dfs 序以及对应子树编号区间
        def dfs(x):
            nonlocal order
            visit[x] = order
            order += 1
            for y in dct[x]:
                if not visit[y]:
                    dfs(y)
            interval[x] = [visit[x], order-1]
            return

        n = len(dct)
        order = 1
        visit = [0]*n
        interval = [[] for _ in range(n)]

        dfs(0)
        return visit, interval

    @staticmethod
    def add_to_n(n):

        # 计算将 [1, 1] 通过 [a, b] 到 [a, a+b] 或者 [a+b, a] 的方式最少次数变成 a == n or b == n
        if n == 1:
            return 0

        def gcd_minus(a, b, c):
            nonlocal ans
            if c >= ans or not b:
                return
            assert a >= b
            if b == 1:
                ans = ans if ans < c + a - 1 else c + a - 1
                return

            # 逆向思维计算保证使 b 减少到 a 以下
            gcd_minus(b, a % b, c + a // b)
            return

        ans = n - 1
        for i in range(1, n):
            gcd_minus(n, i, 0)
        return ans


class TestGeneral(unittest.TestCase):

    def test_dfs(self):
        dfs = DFS()
        dct = [[1, 2], [0, 3], [0, 4], [1], [2]]
        visit, interval = dfs.gen_node_order(dct)
        assert visit == [1, 2, 4, 3, 5]
        assert interval == [[1, 5], [2, 3], [4, 5], [3, 3], [5, 5]]
        return


if __name__ == '__main__':
    unittest.main()
