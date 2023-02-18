import unittest
from collections import deque
from typing import List
from algorithm.src.fast_io import FastIO

"""
算法：广度优先搜索
功能：在有向图与无向图进行扩散，多源双向BFS，0-1BFS（类似SPFA）
题目：

===================================力扣===================================
2493. 将节点分成尽可能多的组（https://leetcode.cn/problems/shortest-palindrome/）利用并查集和广度优先搜索进行连通块分组并枚举最佳方案
2290. 到达角落需要移除障碍物的最小数（https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/）使用0-1 BFS进行优化计算最小代价
1368. 使网格图至少有一条有效路径的最小代价（https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）使用0-1 BFS进行优化计算最小代价
2258. 逃离火灾（https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）使用二分查找加双源BFS进行模拟
2092. 找出知晓秘密的所有专家（https://leetcode.cn/problems/find-all-people-with-secret/）按照时间排序，在同一时间进行BFS扩散

===================================洛谷===================================
P1747 好奇怪的游戏（https://www.luogu.com.cn/problem/P1747）双向BFS搜索最短距离
P5507 机关（https://www.luogu.com.cn/problem/P5507）双向BFS进行搜索
P2040 打开所有的灯（https://www.luogu.com.cn/problem/P2040）定义状态进行 BFS 搜索
P2335 [SDOI2005]位图（https://www.luogu.com.cn/problem/P2335）广度优先搜索
P2385 [USACO07FEB]Bronze Lilypad Pond B（https://www.luogu.com.cn/problem/P2385）广度优先搜索最短步数
P2630 图像变换（https://www.luogu.com.cn/problem/P2630）BFS模拟计算最短次数与最小字典序
P1332 血色先锋队（https://www.luogu.com.cn/problem/P1332）标准BFS
P1330 封锁阳光大学（https://www.luogu.com.cn/problem/P1330）BFS进行隔层染色取较小值，也可以判断连通块是否存在奇数环
P1215 [USACO1.4]母亲的牛奶 Mother's Milk（https://www.luogu.com.cn/problem/P1215）广度优先搜索进行模拟与状态记录
P1037 [NOIP2002 普及组] 产生数（https://www.luogu.com.cn/problem/P1037）广度优先搜索之后进行模拟和枚举
P2853 [USACO06DEC]Cow Picnic S（https://www.luogu.com.cn/problem/P2853）广度优先搜索进行可达计数
P2881 [USACO07MAR]Ranking the Cows G（https://www.luogu.com.cn/problem/P2881）广搜确定已知所有祖先，总共应有n*(n-1)//2对顺序
P2895 [USACO08FEB]Meteor Shower S（https://www.luogu.com.cn/problem/P2895）广度优先搜索模拟
P2960 [USACO09OCT]Invasion of the Milkweed G（https://www.luogu.com.cn/problem/P2960）广度优先搜索裸题
P2298 Mzc和男家丁的游戏（https://www.luogu.com.cn/problem/P2298）BFS裸题
P3139 [USACO16FEB]Milk Pails S（https://www.luogu.com.cn/problem/P3139）广搜加记忆化
P3183 [HAOI2016] 食物链（https://www.luogu.com.cn/problem/P3183）广搜计数计算路径条数，也可以使用深搜DP计数
P4017 最大食物链计数（https://www.luogu.com.cn/problem/P4017）广搜计数计算路径条数，也可以使用深搜DP计数
P3395 路障（https://www.luogu.com.cn/problem/P3395）广度优先搜索进行模拟
P3416 [USACO16DEC]Moocast S（https://www.luogu.com.cn/problem/P3416）广搜加记忆化访问
P3916 图的遍历（https://www.luogu.com.cn/problem/P3916）逆向思维反向建图再加倒序访问传播
P3958 [NOIP2017 提高组] 奶酪（https://www.luogu.com.cn/problem/P3958）建图之后进行广度优先搜索
P4328 [COCI2006-2007#1] Slikar（https://www.luogu.com.cn/problem/P4328）经典广搜题，模拟能否逃离火灾或者洪水
P4961 小埋与扫雷（https://www.luogu.com.cn/problem/P4961）枚举模拟计数，八连通
P6207 [USACO06OCT] Cows on Skates G（https://www.luogu.com.cn/problem/P6207）经典广度优先搜索记录最短路径
P6582 座位调查（https://www.luogu.com.cn/problem/P6582）bfs合法性判断与组合计数快速幂
P7243 最大公约数（https://www.luogu.com.cn/problem/P7243）广度优先搜索加gcd最大公约数计算
P3496 [POI2010]GIL-Guilds（https://www.luogu.com.cn/problem/P3496）脑筋急转弯，BFS隔层染色
P1432 倒水问题（https://www.luogu.com.cn/problem/P1432）经典BFS倒水题，使用记忆化广搜

================================CodeForces================================
E. Nearest Opposite Parity（https://codeforces.com/problemset/problem/1272/E）经典反向建图，多源BFS

参考：OI WiKi（xx）
"""


class BFS:
    def __init__(self):
        return

    @staticmethod
    def cf_1272e(ac=FastIO()):
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = [-1] * n

        # 模板：反向建图
        edge = [[] for _ in range(n)]
        for i in range(n):
            for x in [i + nums[i], i - nums[i]]:
                if 0 <= x < n:
                    edge[x].append(i)

        # 多源 BFS
        for x in [0, 1]:
            stack = [i for i in range(n) if nums[i] % 2 == x]
            visit = set(stack)
            step = 1
            while stack:
                nex = []
                for i in stack:
                    for j in edge[i]:
                        if j not in visit:
                            ans[j] = step
                            nex.append(j)
                            visit.add(j)
                step += 1
                stack = nex
        ac.lst(ans)
        return

    @staticmethod
    def main_3183(n, m, edges):
        # 模板: 计算有向无环图路径条数
        edge = [[] for _ in range(n)]
        degree = [0]*n
        out_degree = [0]*n
        for i, j in edges:
            edge[i].append(j)
            degree[j] += 1
            out_degree[i] += 1
        ind = [i for i in range(n) if degree[i] and not out_degree[i]]
        cnt = [0]*n
        stack = [i for i in range(n) if not degree[i]]
        for x in stack:
            cnt[x] = 1
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    cnt[j] += cnt[i]
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        ans = sum(cnt[i] for i in ind)
        return ans

    @staticmethod
    def bfs_template(grid):
        # 广度优先搜索计算所有 "0" 到最近的 "1" 的距离
        m, n = len(grid), len(grid[0])
        stack = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    grid[i][j] = 0
                    stack.append([i, j])
        # BFS 模板
        step = 1
        while stack:
            nex = []
            for i, j in stack:
                for a, b in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                    if 0 <= a < m and 0 <= b < n and grid[a][b] == "0":
                        grid[a][b] = step
                        nex.append([a, b])
            stack = nex
            step += 1
        return grid

    @staticmethod
    def bilateral_bfs():
        # 双向BFS
        # P1747 好奇怪的游戏
        state = []
        dct = [[0] * 4 for _ in range(12)]
        for i in range(12):
            nums = [int(x) for x in input().strip().split() if x]
            for j in range(1, 5):
                dct[i][j - 1] = nums[j] - 1
            state.append(nums[0] - 1)

        # 定义两个搜索状态
        stack1 = [tuple(state)]
        visit1 = {stack1[0]: tuple()}
        target = tuple([0] * 12)
        flag = (target == stack1[0][0])
        step = 1

        stack2 = [target]
        visit2 = {stack2[0]: tuple()}
        while not flag:
            nex1 = []
            for pre in stack1:
                lst = list(pre)
                for i in range(12):
                    tmp = lst[:]
                    x = dct[i][tmp[i]]
                    tmp[x] += 1
                    tmp[x] %= 4
                    tmp[i] += 1
                    tmp[i] %= 4
                    tmp = tuple(tmp)
                    if tmp not in visit1:
                        nex1.append(tmp)
                        visit1[tmp] = visit1[pre] + (i + 1,)
                        # 判定是否遇见
                        if tmp in visit2:
                            flag = True
                            path = visit1[tmp] + visit2[tmp][::-1]
                            break
                if flag:
                    break

            nex2 = []
            for pre in stack2:
                lst = list(pre)
                for i in range(12):
                    tmp = lst[:]

                    tmp[i] -= 1
                    tmp[i] %= 4

                    x = dct[i][tmp[i]]
                    tmp[x] -= 1
                    tmp[x] %= 4

                    tmp = tuple(tmp)
                    if tmp not in visit2:
                        nex2.append(tmp)
                        visit2[tmp] = visit2[pre] + (i + 1,)
                        # 判定是否遇见
                        if tmp in visit1:
                            flag = True
                            path = visit1[tmp] + visit2[tmp][::-1]
                            break
                if flag:
                    break
            stack1 = nex1
            stack2 = nex2
            step += 1
        print(len(path))
        print(" ".join(str(x) for x in path))
        return

    @staticmethod
    def main_p1747(x0, y0, x2, y2):

        # 双向BFS模板题

        def check(x1, y1):
            if (x1, y1) == (1, 1):
                return 0

            visit1 = {(x1, y1): 0}
            visit2 = {(1, 1): 0}
            direc = [[1, 2], [1, -2], [-1, 2], [-1, -2],
                     [2, 1], [2, -1], [-2, 1], [-2, -1]]
            direc.extend([[2, 2], [2, -2], [-2, 2], [-2, -2]])
            stack1 = [[x1, y1]]
            stack2 = [[1, 1]]
            step = 1

            while True:
                nex1 = []
                for i, j in stack1:
                    for a, b in direc:
                        if 0 < i + a <= 20 and 0 < j + \
                                b <= 20 and (i + a, j + b) not in visit1:
                            visit1[(i + a, j + b)] = step
                            nex1.append([i + a, j + b])
                            if (i + a, j + b) in visit2:
                                return step + visit2[(i + a, j + b)]

                stack1 = nex1

                nex2 = []
                for i, j in stack2:
                    for a, b in direc:
                        if 0 < i + a <= 20 and 0 < j + \
                                b <= 20 and (i + a, j + b) not in visit2:
                            visit2[(i + a, j + b)] = step
                            nex2.append([i + a, j + b])
                            if (i + a, j + b) in visit1:
                                return step + visit1[(i + a, j + b)]

                stack2 = nex2
                step += 1
            return -1

        ans1 = check(x0, y0)
        ans2 = check(x2, y2)
        return ans1, ans2


class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        # L2290
        m, n = len(grid), len(grid[0])
        dis = [[float("inf")] * n for _ in range(m)]
        dis[0][0] = 0
        q = deque([(0, 0)])
        while q:
            x, y = q.popleft()
            for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                if 0 <= nx < m and 0 <= ny < n:
                    g = grid[x][y]
                    if dis[x][y] + g < dis[nx][ny]:
                        dis[nx][ny] = dis[x][y] + g
                        if g == 0:
                            q.appendleft((nx, ny))
                        else:
                            q.append((nx, ny))
        return dis[m - 1][n - 1]


class Solution:
    # L1368
    def minCost(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        BIG = int(1e9)
        dist = [0] + [BIG] * (m * n - 1)
        seen = set()
        import collections
        q = collections.deque([(0, 0)])

        while len(q) > 0:
            x, y = q.popleft()
            if (x, y) in seen:
                continue
            seen.add((x, y))
            cur_pos = x * n + y
            for i, (nx, ny) in enumerate(
                    [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]):
                new_pos = nx * n + ny
                new_dis = dist[cur_pos] + (1 if grid[x][y] != i + 1 else 0)
                if 0 <= nx < m and 0 <= ny < n and new_dis < dist[new_pos]:
                    dist[new_pos] = new_dis
                    if grid[x][y] == i + 1:
                        q.appendleft((nx, ny))
                    else:
                        q.append((nx, ny))

        return dist[m * n - 1]


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        nt = ClassName()
        assert nt.gen_result(10 ** 11 + 131) == 66666666752
        return


if __name__ == '__main__':
    unittest.main()
