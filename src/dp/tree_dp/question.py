


"""
算法：树形DP、树的直径、树上差分、树的重心（以及树的每个节点到其余节点的总距离和）、树的最小偏心距
功能：在树形或者图结构上进行DP，有换根DP，自顶向下和自底向上DP
题目：

===================================力扣===================================
2458 移除子树后的二叉树高度（https://leetcode.cn/problems/height-of-binary-tree-after-subtree-removal-queries/）跑两边DFS进行自顶向下和自底向上DP结合
2440 创建价值相同的连通块（https://leetcode.cn/problems/create-components-with-same-value/）利用总和的因子和树形递归判断连通块是否可行
1569 将子数组重新排序得到同一个二叉查找树的方案数（https://leetcode.cn/problems/number-of-ways-to-reorder-array-to-get-same-bst/solution/by-liupengsay-yi3h/）
968. 监控二叉树（https://leetcode.cn/problems/binary-tree-cameras/）树形DP监控每个节点
2538. 最大价值和与最小价值和的差值（https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/）树形换根DP，求去掉其中一个叶子节点的最大直径
124. 二叉树中的最大路径和（https://leetcode.cn/problems/binary-tree-maximum-path-sum/）树形DP
1617. 统计子树中城市之间最大距离（https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/）二进制枚举加树的直径计算
2378. 选择边来最大化树的得分（https://leetcode.cn/problems/choose-edges-to-maximize-score-in-a-tree/）树形DP
2445. 值为 1 的节点数（https://leetcode.cn/problems/number-of-nodes-with-value-one/）自上而下DP模拟
834. 树中距离之和（https://leetcode.cn/problems/sum-of-distances-in-tree/）树的总距离，求树的重心
1617. 统计子树中城市之间最大距离（https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/）经典枚举直径端点与乘法原理树形DP进行计算
2003. 每棵子树内缺失的最小基因值（https://leetcode.cn/problems/smallest-missing-genetic-value-in-each-subtree/）树形DP启发式合并
2673. 使二叉树所有路径值相等的最小代价（https://leetcode.cn/problems/make-costs-of-paths-equal-in-a-binary-tree/）经典树形DP贪心
1367. 二叉树中的链表（https://leetcode.cn/problems/linked-list-in-binary-tree/description/）典型二叉树与链表比较的记忆化DP
979. 在二叉树中分配硬币（https://leetcode.cn/problems/distribute-coins-in-binary-tree/description/）经典树形DP贪心
1373. 二叉搜索子树的最大键值和（https://leetcode.cn/problems/maximum-sum-bst-in-binary-tree/）经典树形DP二叉树校验
971. 翻转二叉树以匹配先序遍历（https://leetcode.cn/problems/flip-binary-tree-to-match-preorder-traversal/description/）树形DP贪心模拟
100041. 可以到达每一个节点的最少边反转次数（https://www.acwing.com/problem/content/description/4384/）迭代法实现树形换根DP计算，或者一遍DFS或者dfs序加差分
100047. 统计树中的合法路径数目（https://leetcode.cn/problems/count-valid-paths-in-a-tree/description/）树形DP，并查集或者BFS实现

===================================洛谷===================================
P1395 会议（https://www.luogu.com.cn/problem/P1395）树的总距离，求树的重心，单个节点距离其他所有节点的最大距离，换根DP可以做
P1352 没有上司的舞会（https://www.luogu.com.cn/problem/P1352）树形DP，隔层进行动态规划转移
P1922 女仆咖啡厅桌游吧（https://www.luogu.com.cn/problem/P1922）树形DP，贪心进行子树与叶子节点的分配
P2016 战略游戏（https://www.luogu.com.cn/problem/P2016）树形DP瞭望每条边
P1122 最大子树和（https://www.luogu.com.cn/problem/P1122）计算最大的连通块和
P2932 [USACO09JAN]Earthquake Damage G（https://www.luogu.com.cn/problem/P2932）树形DP统计子树个数与贪心安排最小损坏个数
P2996 [USACO10NOV]Visiting Cows G（https://www.luogu.com.cn/problem/P2996）树形DP
P3074 [USACO13FEB]Milk Scheduling S（https://www.luogu.com.cn/problem/P3074）树的最长路径（广搜DP记录最长时间也可以）
P3884 [JLOI2009]二叉树问题（https://www.luogu.com.cn/problem/P3884）基础树形DP计算两点间路径变种长度
P3915 树的分解（https://www.luogu.com.cn/problem/P3915）递归拆解生成等大小的连通块
P4615 [COCI2017-2018#5] Birokracija（https://www.luogu.com.cn/problem/P4615）树形DP
P5002 专心OI - 找祖先（https://www.luogu.com.cn/problem/P5002）使用树形DP与容斥原理进行计数
P5651 基础最短路练习题（https://www.luogu.com.cn/problem/P5651）脑筋急转弯使用并查集去环，转换为树形DP里面任意两点路径的异或和
P6591 [YsOI2020]植树（https://www.luogu.com.cn/problem/P6591）换根DP，即无根树递归判断每个节点作为根节点的情况
P7159 「dWoi R1」Sweet Fruit Chocolate（https://www.luogu.com.cn/problem/P7159）树形DP枚举计数与快速幂计算
P2015 二叉苹果树（https://www.luogu.com.cn/problem/P2015）树形DP，有点像树上背包
P2014 [CTSC1997] 选课（https://www.luogu.com.cn/problem/P2014）树形DP
P4316 绿豆蛙的归宿（https://www.luogu.com.cn/problem/P4316）逆向建图，拓扑排序DP
P1351 [NOIP2014 提高组] 联合权值（https://www.luogu.com.cn/problem/P1351#submit）树形DP
P3304 [SDOI2013]直径（https://www.luogu.com.cn/problem/P3304）经典计算带权无向图的直径以及直径的必经边
P3408 恋爱（https://www.luogu.com.cn/problem/P3408）树形DP
P3478 [POI2008] STA-Station（https://www.luogu.com.cn/problem/P3478）树的质心
P3931 SAC E#1 - 一道难题 Tree（https://www.luogu.com.cn/problem/P3931）典型树形DP
P4084 [USACO17DEC]Barn Painting G（https://www.luogu.com.cn/problem/P4084）典型树形DP
P4395 [BOI2003]Gem 气垫车（https://www.luogu.com.cn/problem/P4395）经典树形 DP 贪心标权值使得整棵树总价值最小
P5765 [CQOI2005]珠宝（https://www.luogu.com.cn/problem/P5765）同P4395
P8602 [蓝桥杯 2013 省 A] 大臣的旅费（https://www.luogu.com.cn/problem/P8602）经典树的直径可用两遍BFS也可用树形DP求解
P8625 [蓝桥杯 2015 省 B] 生命之树（https://www.luogu.com.cn/problem/P8625）树形 DP 典型
P8744 [蓝桥杯 2021 省 A] 左孩子右兄弟（https://www.luogu.com.cn/problem/P8744）简单树形 DP

==================================AtCoder=================================
F - Expensive Expense （https://atcoder.jp/contests/abc222/tasks/abc222_f）换根DP
D. Distance in Tree（https://codeforces.com/problemset/problem/161/D）树形DP计数，记录距离为k的点对数

================================CodeForces================================
C. Uncle Bogdan and Country Happiness（https://codeforces.com/problemset/problem/1388/C）树形DP模拟计算，递归获取子树信息，逆向从上往下还原
F. Maximum White Subtree（https://codeforces.com/problemset/problem/1324/F）经典换根DP题，两遍dfs搜索更新计算
D. Book of Evil（https://codeforces.com/problemset/problem/337/D）经典换根DP题，两遍dfs搜索更新计算
E. Tree Painting（https://codeforces.com/problemset/problem/1187/E）经典换根DP题，两遍dfs搜索更新计算
E. Lomsat gelral（https://codeforces.com/problemset/problem/600/E）迭代方式写深搜序，按秩合并，由小到大
D. A Wide, Wide Graph（https://codeforces.com/problemset/problem/1805/D）树的直径计算，任意点到直径的某个端点的距离最长
G. White-Black Balanced Subtrees（https://codeforces.com/contest/1676/problem/G）使用迭代的方式进行树形DP计算
F. Gardening Friends（https://codeforces.com/contest/1822/problem/F）计算树中节点到其余节点的最大距离
D. Choosing Capital for Treeland（https://codeforces.com/contest/219/problem/D）迭代法实现树形换根DP计算，或者一遍DFS或者dfs序加差分
F. Tree with Maximum Cost（https://codeforces.com/contest/1092/problem/F）带权重树中的总距离，迭代法实现树形换根DP计算
G. Moving to the Capital（https://codeforces.com/contest/1472/problem/G）经典根据最短路进行从下到上与从上到下的DP

================================AcWing================================
3760. 最大剩余油量（https://www.acwing.com/problem/content/description/3763/）脑筋急转弯转化为树形DP迭代方式求解
4381. 翻转树边（https://www.acwing.com/problem/content/description/4384/）迭代法实现树形换根DP计算，或者一遍DFS或者dfs序加差分

参考：OI WiKi（xx）
"""


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1676g(ac=FastIO()):
        # 模板：使用迭代的方式计算树形DP
        for _ in range(ac.read_int()):
            n = ac.read_int()
            parent = ac.read_list_ints_minus_one()
            color = ac.read_str()
            dct = [[] for _ in range(n)]
            for i in range(n - 1):
                dct[parent[i]].append(i + 1)
            ans = 0
            sub = [0] * n
            stack = [[0, 1]]
            while stack:
                i, state = stack.pop()
                if state:
                    stack.append([i, 0])
                    for j in dct[i]:
                        stack.append([j, 1])
                else:
                    x = 0
                    for j in dct[i]:
                        x += sub[j]
                    x += 1 if color[i] == "B" else -1
                    sub[i] = x
                    ans += x == 0
            ac.st(ans)
        return

    @staticmethod
    def cf_1805d(ac=FastIO()):
        # 模板：使用树的直径与端点距离，计算节点对距离至少为k的连通块个数
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)
        tree = TreeDiameterDis(edge)
        u, v = tree.get_diameter_node()
        dis1 = tree.get_bfs_dis(u)
        dis2 = tree.get_bfs_dis(v)
        diff = [0] * n
        for i in range(n):
            diff[ac.max(dis1[i], dis2[i])] += 1
        diff[0] = 1
        diff = list(accumulate(diff, add))
        ac.lst([ac.min(x, n) for x in diff])
        return

    @staticmethod
    def lc_2458(root, queries: List[int]) -> List[int]:
        # 模板：经典类似换根 DP 的思想跑两遍 DFS
        def dfs(node, d):
            if not node:
                return 0
            left = dfs(node.left, d + 1)
            right = dfs(node.right, d + 1)
            h = max(left, right)
            node_depth[node.val] = d
            node_height[node.val] = h
            depth_height[node.val] = d + h
            return h + 1

        # 节点深度
        node_depth = dict()
        # 子树高度
        node_height = dict()
        # 每层节点的子树深度集合
        depth_height = dict()
        dfs(root, 0)

        def get_ans(node, pre):
            if not node:
                return
            ans[node.val] = pre

            pre = max(pre, node_depth[node.val])
            if node.right:
                pre_right = max(pre, depth_height[node.right.val])
            else:
                pre_right = pre

            if node.left:
                pre_left = max(pre, depth_height[node.left.val])
            else:
                pre_left = pre

            get_ans(node.left, pre_right)
            get_ans(node.right, pre_left)
            return

        ans = dict()
        get_ans(root, 0)
        return [ans[q] for q in queries]

    @staticmethod
    def cf_1388c(ac):
        n, m = ac.read_list_ints()
        person = ac.read_list_ints()
        h = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)

        @ac.bootstrap
        def dfs(i, fa):
            nonlocal ans
            a = b = 0
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    a += pos[j]
                    b += neg[j]

            if (h[i] + person[i] + b + a) % 2:
                ans = False
                yield
            good = (h[i] + person[i] + b + a) // 2
            bad = person[i] + a + b - good
            if good < 0 or bad < 0 or bad > person[i] + b:
                ans = False
            pos[i] = good
            neg[i] = bad
            yield

        ans = True
        pos = [0] * n
        neg = [0] * n
        dfs(0, -1)
        return "YES" if ans else "NO"

    @staticmethod
    def cf_161d(n, k, pairs):
        # 模板：记录树中距离为 k 的节点对数量
        edge = [[] for _ in range(n)]
        for x, y in pairs:
            edge[x].append(y)
            edge[y].append(x)
        dp = [[0] * (k + 1) for _ in range(n)]

        def dfs(i, fa):
            nonlocal ans
            dp[i][0] = 1
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    for s in range(1, k + 1):
                        dp[i][s] += dp[j][s - 1]

            ans += dp[i][k]
            cur = 0
            for j in edge[i]:
                if j != fa:
                    for s in range(1, k):
                        cur += dp[j][s - 1] * (dp[i][k - s] - dp[j][k - s - 1])
            ans += cur // 2
            yield

        ans = 0
        dfs(0, -1)
        return ans

    @staticmethod
    def cf_1324f(ac=FastIO()):

        # 模板：换根DP，根据题意进行转换贪心计算结果
        n = ac.read_int()
        nums = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            edge[x].append(y)
            edge[y].append(x)

        @ac.bootstrap
        def dfs(i, fa):
            res = 0
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    cur = son[j] + 2 * nums[j] - 1
                    res += ac.max(cur, 0)
            son[i] = res
            yield

        # 第一遍获取从下往上的最优结果
        son = [0] * n
        dfs(0, -1)

        @ac.bootstrap
        def dfs2(i, fa, pre):
            ans[i] = son[i] + pre + 2 * nums[i] - 1

            lst = [son[j] + 2 * nums[j] - 1 for j in edge[i] if j != fa]
            s = sum([y for y in lst if y >= 0])
            for j in edge[i]:
                if j != fa:
                    tmp = son[j] + 2 * nums[j] - 1
                    cur = ac.max(0, pre + s - ac.max(0, tmp) + 2 * nums[i] - 1)
                    yield dfs2(j, i, cur)
            yield

        # 第二遍获取从下往上的最优结果并更新加和
        ans = [0] * n
        dfs2(0, -1, 0)
        ac.lst(ans)
        return

    @staticmethod
    def cf_337d(ac=FastIO()):
        n, m, d = ac.read_list_ints()
        evil = set(ac.read_list_ints_minus_one())
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        @ac.bootstrap
        def dfs(i, fa):
            res = -inf
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    res = ac.max(res, son[j] + 1)
            if i in evil:
                res = ac.max(res, 0)
            son[i] = res
            yield

        # 计算子节点最远的evil
        son = [-inf] * n
        dfs(0, -1)

        @ac.bootstrap
        def dfs2(i, fa, pre):
            father[i] = pre
            a = b = pre + 1
            for j in edge[i]:
                if j != fa:
                    c = son[j] + 2
                    if c >= a:
                        a, b = c, a
                    elif c >= b:
                        a, b = a, c
            if i in evil:
                c = 1
                if c >= a:
                    a, b = c, a
                elif c >= b:
                    a, b = a, c
            for j in edge[i]:
                if j != fa:
                    c = son[j] + 2
                    if a == c:
                        yield dfs2(j, i, b)
                    else:
                        yield dfs2(j, i, a)
            yield

        # 计算父节点最远的evil
        father = [-inf] * n
        dfs2(0, -1, -inf)
        ans = sum(ac.max(father[i], son[i]) <= d for i in range(n))
        ac.st(ans)
        return

    @staticmethod
    def cf_1092f(ac=FastIO()):
        # 模板：带权重树中的总距离，迭代法实现树形换根DP计算
        n = ac.read_int()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)
        ans = ReRootDP().get_tree_distance_weight(dct, nums)
        ac.st(max(ans))
        return

    @staticmethod
    def lc_968(root: Optional[TreeNode]) -> int:

        # 模板：经典树形DP
        def dfs(node):
            # 不装被监控，装被监控，不装不被监控
            if not node:
                return [0, inf, 0]
            left = dfs(node.left)
            right = dfs(node.right)
            res = [inf, inf, inf]
            res[0] = min(left[1] + min(right[0], right[1]), right[1] + min(left[0], left[1]))
            res[1] = 1 + min(left) + min(right)
            res[2] = left[0] + right[0]
            return res

        ans = dfs(root)
        return min(ans[0], ans[1])

    @staticmethod
    def lc_1367(head: Optional[ListNode], root: Optional[TreeNode]) -> bool:

        # 模板：典型二叉树与链表比较的记忆化DP

        @lru_cache(None)
        def dfs(lst, node):
            if not lst:
                return True
            if not node:
                return False
            if lst.val == node.val:
                if dfs(lst.next, node.left) or dfs(lst.next, node.right):
                    return True
            return dfs(head, node.left) or dfs(head, node.right)

        return dfs(head, root)

    @staticmethod
    def lc_1617(n: int, edges: List[List[int]]) -> List[int]:
        # 模板：枚举子集使用并查集判断连通性再计算树的直径
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
                    dct[ind[u]].append(ind[v])
                    dct[ind[v]].append(ind[u])
                    uf.union(ind[u], ind[v])
            if uf.part != 1:
                continue
            # 计算直径或者是get_diameter_bfs都可以
            ans[TreeDiameter().get_diameter_dfs(dct)] += 1
        return ans[1:]

    @staticmethod
    def cf_1187e(ac=FastIO()):
        # 模板：经典换根DP题计算最佳结果
        n = ac.read_int()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        @ac.bootstrap
        def dfs(i, fa):
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    down[i] += down[j]
                    son[i] += son[j]
            son[i] += 1
            down[i] += son[i]
            yield

        down = [0] * n
        son = [0] * n
        dfs(0, -1)

        @ac.bootstrap
        def dfs2(i, fa, pre):
            up[i] = pre
            res = sum(down[j] for j in edge[i] if j != fa)
            for j in edge[i]:
                if j != fa:
                    yield dfs2(j, i, (n - son[j]) + pre + (res - down[j]))
            yield

        up = [0] * n
        dfs2(0, -1, 0)
        ac.st(max(up[i] + (down[i] - son[i]) + n for i in range(n)))
        return

    @staticmethod
    def cf_600e_bfs(ac=FastIO()):
        # 模板：自下而上递归的迭代写法，从小到大按秩合并
        n = ac.read_int()
        colors = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            edge[i].append(j)
            edge[j].append(i)
        # 深搜序自下而上以及父子信息获取
        stack = [[0, -1]]
        parent = [-1] * n
        down_to_up_order = []
        while stack:
            i, fa = stack.pop()
            down_to_up_order.append(i)
            for j in edge[i]:
                if j != fa:
                    stack.append([j, i])
                    parent[j] = i
        down_to_up_order.reverse()

        # 维护一个最大值的出现次数
        mx = [0] * n
        ans = [0] * n
        dp = [None] * n
        for i in down_to_up_order:
            dp[i] = Counter()
            dp[i][colors[i]] += 1
            mx[i] = 1
            ans[i] = colors[i]
            for j in edge[i]:
                if dp[j]:
                    if len(dp[j]) > len(dp[i]):
                        # 从小到大
                        dp[i], dp[j] = dp[j], dp[i]
                        mx[i] = mx[j]
                        ans[i] = ans[j]
                    for w in dp[j]:
                        # 按秩合并
                        dp[i][w] += dp[j][w]
                        if dp[i][w] == mx[i]:
                            ans[i] += w
                        elif dp[i][w] > mx[i]:
                            mx[i] = dp[i][w]
                            ans[i] = w
                    # 及时清空
                    dp[j] = None
        ac.lst(ans)
        return

    @staticmethod
    def cf_600e_dfs(ac=FastIO()):
        # 模板：自下而上递归的递归写法，从小到大按秩合并
        n = ac.read_int()
        nums = ac.read_list_ints()
        edge = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v = ac.read_list_ints_minus_one()
            edge[u].append(v)
            edge[v].append(u)

        @ac.bootstrap
        def dfs(i, fa):
            dp[i] = Counter()
            dp[i][nums[i]] += 1
            ceil[i] = 1
            cur = nums[i]
            for j in edge[i]:
                if j != fa:
                    yield dfs(j, i)
                    if len(dp[j]) > len(dp[i]):
                        dp[i], dp[j] = dp[j], dp[i]
                        cur = ans[j]
                        ceil[i] = ceil[j]
                    for num in dp[j]:
                        dp[i][num] += dp[j][num]
                        if dp[i][num] > ceil[i]:
                            ceil[i] = dp[i][num]
                            cur = num
                        elif dp[i][num] == ceil[i]:
                            cur += num
                    dp[j] = None
            ans[i] = cur
            yield

        dp = [None] * n
        ans = [0] * n
        ceil = [0] * n
        dfs(0, -1)
        ac.lst(ans)
        return

    @staticmethod
    def lg_p1395_1(ac=FastIO()):
        # 模板：计算树的重心为最大子树节点数最小
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints()
            dct[i - 1].append(j - 1)
            dct[j - 1].append(i - 1)

        root = ReRootDP().get_tree_centroid(dct)

        def bfs_diameter(src):
            ans = 0
            stack = [[src, 0]]
            parent = [-1] * n
            while stack:
                u, dis = stack.pop()
                ans += dis
                for v in dct[u]:
                    if v != parent[u]:
                        parent[v] = u
                        stack.append([v, dis + 1])
            return ans

        ac.lst([root + 1, bfs_diameter(root)])
        return

    @staticmethod
    def lg_p1395_2(ac=FastIO()):
        # 模板：计算树的重心为距离其余所有节点
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints()
            dct[i - 1].append(j - 1)
            dct[j - 1].append(i - 1)

        ans = ReRootDP().get_tree_distance(dct)
        dis = min(ans)
        ac.lst([ans.index(dis) + 1, dis])
        return

    @staticmethod
    def cf_1822f(ac=FastIO()):
        # 模板：换根 DP 计算树中节点其余节点最大的距离
        for _ in range(ac.read_int()):
            n, k, c = ac.read_list_ints()
            dct = [[] for _ in range(n)]
            for _ in range(n - 1):
                i, j = ac.read_list_ints_minus_one()
                dct[i].append(j)
                dct[j].append(i)

            dis = ReRootDP().get_tree_distance_max(dct)

            ans = -inf
            stack = [[0, 0, -1]]
            while stack:
                i, d, fa = stack.pop()
                cur = dis[i] * k - d
                ans = ac.max(ans, cur)
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, d + c, i])
            ac.st(ans)
        return

    @staticmethod
    def lg_p1352(ac=FastIO()):
        # 模板：使用树形DP的迭代写法进行计算
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]

        dct = [[] for _ in range(n)]
        degree = [0] * n
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            dct[y].append(x)
            degree[x] += 1

        root = [i for i in range(n) if not degree[i]][0]
        dp = [[0, 0] for _ in range(n)]
        stack = [[root, -1]]
        while stack:
            # 为取反码后的负数则直接出栈
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                x = nums[i] if nums[i] > 0 else 0
                y = 0
                for j in dct[i]:
                    if j != fa:
                        a, b = dp[j]
                        x += a
                        y += b
                dp[i] = [y, ac.max(x, y)]
        ac.st(max(dp[root]))
        return

    @staticmethod
    def lg_p2015(ac=FastIO()):
        # 模板：树形DP
        n, q = ac.read_list_ints()
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            x, y, z = ac.read_list_ints()
            dct[x - 1][y - 1] = z
            dct[y - 1][x - 1] = z
        dp = [[0] * (q + 1) for _ in range(n)]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                if len(dct[i]) > 1:
                    a, b = [x for x in dct[i] if x != fa]
                    for j in range(1, q + 1):
                        cur = ac.max(dp[a][j - 1] + dct[i][a], dp[b][j - 1] + dct[i][b])
                        for k in range(j - 1):
                            cur = ac.max(cur, dp[a][k] + dp[b][j - k - 2] + dct[i][a] + dct[i][b])
                        dp[i][j] = cur
        ac.st(dp[0][q])
        return

    @staticmethod
    def lg_p2014(ac=FastIO()):
        # 模板：树形DP加背包DP
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n + 1)]
        nums = [0]
        for i in range(n):
            k, s = ac.read_list_ints()
            nums.append(s)
            dct[k].append(i + 1)
        dp = [[0] * (m + 2) for _ in range(n + 1)]
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                dp[i][1] = nums[i]
                for j in dct[i]:
                    if j != fa:
                        cur = dp[i][:]
                        for x in range(1, m + 2):
                            for y in range(m + 2 - x):
                                cur[x + y] = ac.max(cur[x + y], dp[i][x] + dp[j][y])
                        dp[i] = cur[:]
        ac.st(dp[0][m + 1])
        return

    @staticmethod
    def lg_p4316(ac=FastIO()):
        # 模板：反向建图加拓扑排序树形概率DP
        n, m = ac.read_list_ints()
        dp = [0 for _ in range(n)]
        degree = [0] * n
        dct = [dict() for _ in range(n)]
        for _ in range(m):
            a, b, w = ac.read_list_ints()
            a -= 1
            b -= 1
            dct[b][a] = w
            degree[a] += 1
        cnt = degree[:]

        stack = deque([n - 1])
        while stack:
            i = stack.popleft()
            k = len(dct[i])
            a = dp[i]
            for j in dct[i]:
                dp[j] += a + dct[i][j]
                degree[j] -= 1
                if not degree[j]:
                    dp[j] /= cnt[j]
                    stack.append(j)
        ans = "%.2f" % (dp[0])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1351(ac=FastIO()):
        # 模板：树形DP
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        nums = ac.read_list_ints()
        ceil = ans = 0
        mod = 10007
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            lst = []
            if fa != -1:
                lst.append(nums[fa])
            for j in dct[i]:
                if j != fa:
                    stack.append([j, i])
                    lst.append(nums[j])
            if lst:

                s = sum(lst)
                a = b = 0
                for num in lst:
                    ans += num * (s - num)
                    ans %= mod
                    if num > a:
                        a, b = num, a
                    elif num > b:
                        b = num
                ceil = ac.max(ceil, a * b)

        ac.lst([ceil, ans])
        return

    @staticmethod
    def lg_p3304(ac=FastIO()):
        # 模板：经典计算带权无向图的直径以及直径的必经边
        n = ac.read_int()
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, k = ac.read_list_ints()
            i -= 1
            j -= 1
            dct[i][j] = dct[j][i] = k
        # 首先计算直径
        tree = TreeDiameterWeighted()
        x, _, _ = tree.bfs(dct, 0)
        y, path, dia = tree.bfs(dct, x)
        ac.st(dia)
        # 确定直径上每个点的最远端距离
        nodes = set(path)
        dis = [0] * n
        for x in path:
            q = [[x, -1, 0]]
            while q:
                i, fa, d = q.pop()
                for j in dct[i]:
                    if j != fa and j not in nodes:
                        dis[x] = d + dct[i][j]
                        q.append([j, i, d + dct[i][j]])

        # 计算直径必经边的最右边端点
        m = len(path)
        pre = right = 0
        for j in range(1, m):
            pre += dct[path[j - 1]][path[j]]
            right = j
            if dis[path[j]] == dia - pre:  # 此时点下面有非当前直径的最远路径
                break

        # 计算直径必经边的最左边端点
        left = m - 1
        post = 0
        for j in range(m - 2, -1, -1):
            post += dct[path[j]][path[j + 1]]
            left = j
            if dis[path[j]] == dia - post:  # 此时点下面有非当前直径的最远路径
                break

        ans = ac.max(0, right - left)
        ac.st(ans)
        return

    @staticmethod
    def lg_3408(ac=FastIO()):

        # 模板：树形 DP 模拟
        n, t, c = ac.read_list_ints()
        dct = [[] for _ in range(n + 1)]
        nums = [c]
        for i in range(n):
            b, a = ac.read_list_ints()
            dct[b].append(i + 1)
            nums.append(a)

        stack = [0]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                if not dct[i]:
                    continue
                else:
                    # 收到子树下属的最少花费
                    if nums[i] > t:
                        continue

                    # 需要最少的 x 个下属花费
                    x = math.ceil(len(dct[i]) * nums[i] / t)
                    lst = []
                    for j in dct[i]:
                        lst.append(nums[j])
                    lst.sort()
                    nums[i] = sum(lst[:x])
        ac.st(nums[0])
        return

    @staticmethod
    def lg_p3478(ac=FastIO()):
        # 模板：计算树的质心
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        dis = ReRootDP().get_tree_distance(dct)
        ind = 0
        for i in range(1, n):
            if dis[i] > dis[ind]:
                ind = i
        ac.st(ind + 1)
        return

    @staticmethod
    def lg_p3931(ac=FastIO()):
        # 模板：树形 DP 模拟
        n, root = ac.read_list_ints()
        root -= 1
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, c = ac.read_list_ints_minus_one()
            c += 1
            dct[i][j] = dct[j][i] = c
        stack = [[root, -1]]
        sub = [inf] * n
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                if len(dct[i]) == 1 and i != root:
                    continue
                res = 0
                for j in dct[i]:
                    if j != fa:
                        res += ac.min(dct[i][j], sub[j])
                sub[i] = res
        ac.st(sub[root] if sub[root] < inf else 0)
        return

    @staticmethod
    def lg_p4395(ac=FastIO()):
        # 模板：经典树形 DP 贪心标权值使得整棵树总价值最小
        n = ac.read_int()
        ceil = int(math.log2(n)) + 1
        sub = [[inf] * (ceil + 1) for _ in range(n)]
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)

        # 迭代写法
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                cur = [0] * (ceil + 1)
                for j in dct[i]:
                    if j != fa:
                        # 记录子树最小的两个值
                        a = b = inf
                        for c in sub[j][1:]:
                            if c < a:
                                a, b = c, a
                            elif c < b:
                                b = c
                        # 记录 i 赋值为 x 时的子树价值和
                        for x in range(1, ceil + 1):
                            if sub[j][x] == a:
                                cur[x] += b
                            else:
                                cur[x] += a
                for x in range(1, ceil + 1):
                    sub[i][x] = x + cur[x]
        ac.st(min(sub[0][1:]))
        return

    @staticmethod
    def lg_p8625(ac=FastIO()):
        # 模板：树形 DP 典型
        n = ac.read_int()
        nums = ac.read_list_ints()
        sub = [0] * n
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                res = 0
                for j in dct[i]:
                    if j != fa:
                        res += sub[j]
                res += nums[i]
                sub[i] = res if res > 0 else 0
        ac.st(max(sub))
        return

    @staticmethod
    def lc_1617_2(n: int, edges: List[List[int]]) -> List[int]:
        # 模板：经典枚举直径端点与乘法原理树形DP进行计算
        dct = [[] for _ in range(n)]
        for i, j in edges:
            i -= 1
            j -= 1
            dct[i].append(j)
            dct[j].append(i)

        dis = []
        for i in range(n):
            cur = [inf] * n
            cur[i] = 0
            stack = deque([i])
            while stack:
                i = stack.pop()
                for j in dct[i]:
                    if cur[j] == inf:
                        cur[j] = cur[i] + 1
                        stack.append(j)
            dis.append(cur[:])

        ans = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = dis[i][j]

                stack = [[i, -1]]
                sub = [0] * n
                while stack:
                    ii, fa = stack.pop()
                    if ii >= 0:
                        stack.append([~ii, fa])
                        for jj in dct[ii]:
                            if jj != fa:
                                if (dis[i][jj] < d or (dis[i][jj] == d and jj > j)) and \
                                        (dis[j][jj] < d or (dis[j][jj] == d and jj > i)):
                                    stack.append([jj, ii])
                    else:
                        ii = ~ii
                        sub[ii] = 1  # x是必点
                        for jj in dct[ii]:
                            if jj != fa:
                                if (dis[i][jj] < d or (dis[i][jj] == d and jj > j)) and \
                                        (dis[j][jj] < d or (dis[j][jj] == d and jj > i)):
                                    sub[ii] *= sub[jj]
                        if dis[i][ii] + dis[j][ii] > d:  # x 是可选点
                            sub[ii] += 1
                ans[d] += sub[i]

        return ans[1:]

    @staticmethod
    def ac_3760(ac=FastIO()):
        # 模板：脑筋急转弯转化为树形DP迭代方式求解
        n = ac.read_int()
        w = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v, c = ac.read_list_ints()
            u -= 1
            v -= 1
            dct[u].append([v, c])
            dct[v].append([u, c])
        ans = 0

        stack = [[0, -1]]
        sub = [0 for _ in range(n)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j, cc in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i

                d1, d2 = 0, 0
                for j, cc in dct[i]:
                    if j != fa:
                        d = sub[j] - cc
                        if d >= d1:
                            d1, d2 = d, d1
                        elif d >= d2:
                            d2 = d
                if d1 + d2 + w[i] > ans:
                    ans = d1 + d2 + w[i]
                sub[i] = d1 + w[i]
        ac.st(ans)
        return

    @staticmethod
    def ac_4381(ac=FastIO()):
        # 模板：迭代法实现树形换根DP计算
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append([y, 0])
            dct[y].append([x, 1])

        # 第一遍DP计算子节点的影响，从上到下再从下到上累加
        sub = [0] * n
        stack = [[0, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append([~i, fa])
                for j, w in dct[i]:
                    if j != fa:
                        stack.append([j, i])
            else:
                i = ~i
                cur = 0
                for j, w in dct[i]:
                    if j != fa:
                        cur += sub[j] + w
                sub[i] = cur

        # 第二遍DP计算祖先节点与兄弟节点的影响
        cnt = [0] * n
        stack = [[0, -1, 0]]
        while stack:
            i, fa, pre = stack.pop()
            cur = sub[i]
            sub[i] += pre
            for j, w in dct[i]:
                if j != fa:
                    stack.append([j, i, pre + cur - sub[j] - w + 1 - w])

        x = min(sub)
        ac.st(x)
        res = [i + 1 for i in range(n) if sub[i] == x]
        ac.lst(res)
        return

    @staticmethod
    def lc_100041(n: int, edges: List[List[int]]) -> List[int]:
        # 模板：一遍DFS迭代实现树形换根DP计算
        dct = [[] for _ in range(n)]
        for x, y in edges:
            dct[x].append([y, 1])
            dct[y].append([x, 0])

        sub_cnt = [0] * n
        sub_one = [0] * n
        pre_cnt = [0] * n
        pre_one = [0] * n
        stack = [[0, -1]]
        while stack:
            x, fa = stack.pop()
            if x >= 0:
                stack.append([~x, fa])
                for y, w in dct[x]:
                    if y != fa:
                        pre_cnt[y] = pre_cnt[x] + 1
                        pre_one[y] = pre_one[x] + w
                        stack.append([y, x])
            else:
                x = ~x
                sub_cnt[x] = 1
                for y, w in dct[x]:
                    if y != fa:
                        sub_cnt[x] += sub_cnt[y]
                        sub_one[x] += sub_one[y] + w
        ans = [pre_one[i] + (sub_cnt[i] - sub_one[i]) + (
                    n - 1 - pre_cnt[i] - sub_cnt[i] - (sub_one[0] - sub_one[i] - pre_one[i])) for i in range(n)]
        return ans

    @staticmethod
    def lc_2673(n: int, cost: List[int]) -> int:
        # 模板：经典树形DP贪心
        ans = 0
        for i in range(n // 2, 0, -1):
            left = cost[i * 2 - 1]
            right = cost[i * 2]
            if left > right:
                cost[i - 1] += left
                ans += left - right
            else:
                cost[i - 1] += right
                ans += right - left
        return ans
