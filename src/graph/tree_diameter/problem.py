"""
Algorithm：广度优先搜索、双端队列BFS、离散化BFS、有边界的BFS、染色法、奇数环
Function：在有向图与无向图扩散，多源BFS、双向BFS，0-1BFS（类似SPFA）双向BFS或者A-star启发式搜索

====================================LeetCode====================================
1036（https://leetcode.com/problems/escape-a-large-maze/）带边界的BFS和离散化BFS两种解法
2493（https://leetcode.com/problems/divide-nodes-into-the-maximum-number-of-groups/）利用union_find和广度优先搜索连通块分组并brute_force最佳方案，也就是染色法判断是否可以形成bipartite_graph
2290（https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/）0-1 BFS优化最小代价
1368（https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）0-1 BFS优化最小代价
2258（https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/）binary_search|双源BFSimplemention
2092（https://leetcode.com/problems/find-all-people-with-secret/）按照时间sorting，在同一时间BFS扩散
2608（https://leetcode.com/contest/biweekly-contest-101/problems/shortest-cycle-in-a-graph/）BFS求无向图的最短环，还可以删除边两点最短路成为环，或者以任意边为起点，逐渐|边
1197（https://leetcode.com/problems/minimum-knight-moves/?envType=study-plan-v2&id=premium-algo-100）双向BFS，或者BFS确定边界
1654（https://leetcode.com/problems/minimum-jumps-to-reach-home/）BFS，证明确定上界implemention
1926（https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/）双端队列01BFS原地hash
909（https://leetcode.com/problems/snakes-and-ladders/）01BFSimplemention
1210（https://leetcode.com/problems/minimum-moves-to-reach-target-with-rotations/description/）01BFSimplemention
1298（https://leetcode.com/problems/maximum-candies-you-can-get-from-boxes/）BFS
928（https://leetcode.com/problems/minimize-malware-spread-ii/description/）brute_force起始点BFS
994（https://leetcode.com/problems/rotting-oranges/description/）BFS队列implemention

=====================================LuoGu======================================
1747（https://www.luogu.com.cn/problem/P1747）双向BFS搜索最短距离
5507（https://www.luogu.com.cn/problem/P5507）双向BFS搜索
2040（https://www.luogu.com.cn/problem/P2040）定义状态 BFS 搜索
2335（https://www.luogu.com.cn/problem/P2335）广度优先搜索
2385（https://www.luogu.com.cn/problem/P2385）广度优先搜索最短步数
2630（https://www.luogu.com.cn/problem/P2630）BFSimplemention最短次数与最小lexicographical_order
1332（https://www.luogu.com.cn/problem/P1332）标准BFS
1330（https://www.luogu.com.cn/problem/P1330）BFS隔层染色法取较小值，也可以判断连通块是否存在奇数环
1215（https://www.luogu.com.cn/problem/P1215）广度优先搜索implemention与状态记录
1037（https://www.luogu.com.cn/problem/P1037）广度优先搜索之后implemention和brute_force
2853（https://www.luogu.com.cn/problem/P2853）广度优先搜索可达counter
2881（https://www.luogu.com.cn/problem/P2881）广搜确定已知所有祖先，总共应有n*(n-1)//2对顺序
2895（https://www.luogu.com.cn/problem/P2895）广度优先搜索implemention
2960（https://www.luogu.com.cn/problem/P2960）广度优先搜索裸题
2298（https://www.luogu.com.cn/problem/P2298）BFS裸题
3139（https://www.luogu.com.cn/problem/P3139）广搜|记忆化
3183（https://www.luogu.com.cn/problem/P3183）广搜counter路径条数，也可以深搜DPcounter
4017（https://www.luogu.com.cn/problem/P4017）广搜counter路径条数，也可以深搜DPcounter
3395（https://www.luogu.com.cn/problem/P3395）广度优先搜索implemention
3416（https://www.luogu.com.cn/problem/P3416）广搜|记忆化访问
3916（https://www.luogu.com.cn/problem/P3916）reverse_thinking反向建图再|倒序访问传播
3958（https://www.luogu.com.cn/problem/P3958）建图之后广度优先搜索
4328（https://www.luogu.com.cn/problem/P4328）广搜题，implemention能否逃离火灾或者洪水
4961（https://www.luogu.com.cn/problem/P4961）brute_forceimplementioncounter，八连通
6207（https://www.luogu.com.cn/problem/P6207）广度优先搜索记录最短路径
6582（https://www.luogu.com.cn/problem/P6582）bfs合法性判断与组合counter快速幂
7243（https://www.luogu.com.cn/problem/P7243）广度优先搜索|gcd最大公约数
3496（https://www.luogu.com.cn/problem/P3496）brain_teaser，BFS隔层染色
1432（https://www.luogu.com.cn/problem/P1432）BFS倒水题，记忆化广搜
1807（https://www.luogu.com.cn/problem/P1807）不保证连通的有向无环图求 1 到 n 的最长路
1379（https://www.luogu.com.cn/problem/P1379）双向BFS
5507（https://www.luogu.com.cn/problem/P5507）双向BFS或者A-star启发式搜索
5908（https://www.luogu.com.cn/problem/P5908）无根树直接bfs遍历
1099（https://www.luogu.com.cn/problem/P1099）题，用到了树的直径、BFS、two_pointer和单调队列求最小偏心距
2491（https://www.luogu.com.cn/problem/P2491）同树网的核P1099
1038（https://www.luogu.com.cn/problem/P1038）拓扑sorting题
1126（https://www.luogu.com.cn/problem/P1126）广度优先搜索
1213（https://www.luogu.com.cn/problem/P1213）state_compression优化01BFS
1902（https://www.luogu.com.cn/problem/P1902）binary_search|BFS与原地hash路径最大值的最小值
2199（https://www.luogu.com.cn/problem/P2199）队列01BFS判定距离最近的可视范围
2226（https://www.luogu.com.cn/problem/P2226）有限制地BDS转向
2296（https://www.luogu.com.cn/problem/P2296）正向与反向建图跑两次BFS
2919（https://www.luogu.com.cn/problem/P2919）bfs按元素值sorting后从大到小遍历
2937（https://www.luogu.com.cn/problem/P2937）01BFSpriority_queue
3456（https://www.luogu.com.cn/problem/P3456） BFS 与周边山峰山谷
3496（https://www.luogu.com.cn/problem/P3496）brain_teaser| BFS 
3818（https://www.luogu.com.cn/problem/P3818）队列 01BFS 状态广搜
3855（https://www.luogu.com.cn/problem/P3855）定义四维状态的广度优先搜索
3869（https://www.luogu.com.cn/problem/P3869）广搜|状压记录最少次数
4554（https://www.luogu.com.cn/problem/P4554）典型 01BFS implemention
4667（https://www.luogu.com.cn/problem/P4667） 01BFS implemention
5096（https://www.luogu.com.cn/problem/P5096）状压|广搜 BFS implemention
5099（https://www.luogu.com.cn/problem/P5099）队列 01BFS 广搜implemention
5195（https://www.luogu.com.cn/problem/P5195）
6131（https://www.luogu.com.cn/problem/P6131） BFS 不同连通块之间的距离
6909（https://www.luogu.com.cn/problem/P6909）预处理| BFS 
8628（https://www.luogu.com.cn/problem/P8628）简单 01 BFS 
8673（https://www.luogu.com.cn/problem/P8673）简单 01 BFS implemention
8674（https://www.luogu.com.cn/problem/P8674）预处理建图后 BFS implemention
9065（https://www.luogu.com.cn/problem/P9065）brain_teaserBFSbrute_force

===================================CodeForces===================================
1272E（https://codeforces.com/problemset/problem/1272/E）反向建图，多源BFS
1572A（https://codeforces.com/problemset/problem/1572/A）brain_teaser建图，广度优先搜索是否存在环与无环时从任意起点的DAG最长路
1037D（https://codeforces.com/problemset/problem/1037/D）BDS好题，结合队列与集合implemention

====================================AtCoder=====================================
D - People on a Line（https://atcoder.jp/contests/abc087/tasks/arc090_b）BFS判断类差分约束问题，差分约束问题复杂度O(n^2)，本题1e5的等式BFS
E - Virus Tree 2（https://atcoder.jp/contests/abc133/tasks/abc133_e）BFS染色法counter

=====================================AcWing=====================================
173（https://www.acwing.com/problem/content/175/）多源BFS模板题
175（https://www.acwing.com/problem/content/177/）双端priority_queue BFS
177（https://www.acwing.com/problem/content/179/）多源双向BFS
4415（https://www.acwing.com/problem/content/description/4418）BFS染色法，判断有无奇数环，方案counter
4481（https://www.acwing.com/problem/content/description/4484/）01BFS

"""

from typing import List

from src.graph.tree_diameter.template import TreeDiameter
from src.graph.union_find.template import UnionFind
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1805d(ac=FastIO()):
        # 树的直径与端点距离，节点对距离至少为k的连通块个数
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append([v, 1])
            edge[v].append([u, 1])
        tree = TreeDiameter(edge)
        u, v = tree.get_diameter_info()[:2]
        dis1, _ = tree.get_bfs_dis(u)
        dis2, _ = tree.get_bfs_dis(v)
        diff = [0] * n
        for i in range(n):
            diff[ac.max(dis1[i], dis2[i])] += 1
        diff[0] = 1
        diff = ac.accumulate(diff)[1:]
        ac.lst([ac.min(x, n) for x in diff])
        return

    @staticmethod
    def lg_p3304(ac=FastIO()):
        # 带权无向图的直径以及直径的必经边
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        original = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, k = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i].append([j, k])
            dct[j].append([i, k])
            original[i][j] = original[j][i] = k
        # 首先直径
        tree = TreeDiameter(dct)
        x, y, path, dia = tree.get_diameter_info()
        ac.st(dia)
        # 确定直径上每个点的最远端距离
        nodes = set(path)
        dis = [0] * n
        for x in path:
            q = [[x, -1, 0]]
            while q:
                i, fa, d = q.pop()
                for j, w in dct[i]:
                    if j != fa and j not in nodes:
                        dis[x] = d + w
                        q.append([j, i, d + w])

        # 直径必经边的最右边端点
        m = len(path)
        pre = right = 0
        for j in range(1, m):
            pre += original[path[j - 1]][path[j]]
            right = j
            if dis[path[j]] == dia - pre:  # 此时点下面有非当前直径的最远路径
                break

        # 直径必经边的最左边端点
        left = m - 1
        post = 0
        for j in range(m - 2, -1, -1):
            post += original[path[j]][path[j + 1]]
            left = j
            if dis[path[j]] == dia - post:  # 此时点下面有非当前直径的最远路径
                break

        ans = ac.max(0, right - left)
        ac.st(ans)
        return

    @staticmethod
    def lc_1617(n: int, edges: List[List[int]]) -> List[int]:
        # brute_force子集union_find判断连通性再树的直径
        ans = [0] * n
        for state in range(1, 1 << n):
            node = [i for i in range(n) if state & (1 << i)]
            ind = {num: i for i, num in enumerate(node)}
            m = len(node)
            dct = [[] for _ in range(m)]
            uf = UnionFind(m)
            for u, v in edges:
                u -= 1
                v -= 1
                if u in ind and v in ind:
                    dct[ind[u]].append([ind[v], 1])
                    dct[ind[v]].append([ind[u], 1])
                    uf.union(ind[u], ind[v])
            if uf.part != 1:
                continue
            tree = TreeDiameter(dct)
            ans[tree.get_diameter_info()[-1]] += 1
        return ans[1:]

    @staticmethod
    def lq_5890(ac=FastIO()):
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        ans = 0
        for _ in range(n - 1):
            u, v, w = ac.read_list_ints()
            u -= 1
            v -= 1
            dct[u].append([v, w])
            dct[v].append([u, w])
            ans += w * 2
        dis = TreeDiameter(dct).get_diameter_info()[-1]
        ans -= dis
        ac.st(ans)
        return