"""
算法：树状数组、二维树状数组
功能：进行数组区间加减，和区间值求和（单点可转换为区间）
题目：

===================================力扣===================================
307. 区域和检索 - 数组可修改（https://leetcode.cn/problems/range-sum-query-mutable）PointChangeRangeSum
1409. 查询带键的排列（https://leetcode.cn/problems/queries-on-a-permutation-with-key/）经典树状数组模拟
1626. 无矛盾的最佳球队（https://leetcode.cn/problems/best-team-with-no-conflicts/）树状数组维护前缀最大值，也可使用动态规划求解
6353. 网格图中最少访问的格子数（https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/）树状数组维护前缀区间最小值单点更新
308. 二维区域和检索 - 可变（https://leetcode.cn/problems/range-sum-query-2d-mutable/）二维树状数组，单点增减与区间和查询
2659. 将数组清空（https://leetcode.cn/problems/make-array-empty/submissions/）经典模拟删除，可以使用树状数组也可以使用SortedList也可以使用贪心
1505. 最多 K 次交换相邻数位后得到的最小整数（https://leetcode.cn/problems/minimum-possible-integer-after-at-most-k-adjacent-swaps-on-digits/）经典树状数组模拟计数移动，也可以使用SortedList
2193. 得到回文串的最少操作次数（https://leetcode.cn/problems/minimum-number-of-moves-to-make-palindrome/description/）使用树状数组贪心模拟交换构建回文串，相同题目（P5041求回文串）
2407. 最长递增子序列 II（https://leetcode.cn/problems/longest-increasing-subsequence-ii/description/）树状数组加线性DP
100112. 平衡子序列的最大和（https://leetcode.cn/problems/maximum-balanced-subsequence-sum/）离散化树状数组加线性DP
2736. 最大和查询（https://leetcode.cn/problems/maximum-sum-queries/）PointAddPreMax

===================================洛谷===================================
P2068 统计和（https://www.luogu.com.cn/problem/P2068）单点更新与区间求和
P2345 [USACO04OPEN] MooFest G（https://www.luogu.com.cn/problem/P2345）使用两个树状数组计数与加和更新查询
P2357 守墓人（https://www.luogu.com.cn/problem/P2357）区间更新与区间求和
P2781 传教（https://www.luogu.com.cn/problem/P2781）区间更新与区间求和
P5200 [USACO19JAN]Sleepy Cow Sorting G（https://www.luogu.com.cn/problem/P5200）树状数组加贪心模拟
P3374 树状数组 1（https://www.luogu.com.cn/problem/P3374）区间值更新与求和
P3368 树状数组 2（https://www.luogu.com.cn/problem/P3368）区间值更新与求和
P5677 配对统计（https://www.luogu.com.cn/problem/P5677）区间值更新与求和
P5094 [USACO04OPEN] MooFest G 加强版（https://www.luogu.com.cn/problem/P5094）单点更新增加值与前缀区间和查询
P1816 忠诚（https://www.luogu.com.cn/problem/P1816）树状数组查询静态区间最小值
P1908 逆序对（https://www.luogu.com.cn/problem/P1908）树状数组求逆序对
P1725 琪露诺（https://www.luogu.com.cn/problem/P1725）倒序线性DP，单点更新值，查询区间最大值
P3586 [POI2015] LOG（https://www.luogu.com.cn/problem/P3586）离线查询、离散化树状数组，单点增减，前缀和查询
P1198 [JSOI2008] 最大数（https://www.luogu.com.cn/problem/P1198）树状数组，查询区间最大值
P4868 Preprefix sum（https://www.luogu.com.cn/problem/P4868）经典转换公式单点修改，使用两个树状数组维护前缀和的前缀和
P5463 小鱼比可爱（加强版）（https://www.luogu.com.cn/problem/P5463）经典使用树状数组维护前缀计数，枚举最大值计算所有区间数贡献
P6225 [eJOI2019] 异或橙子（https://www.luogu.com.cn/problem/P6225）经典使用树状数组维护前缀异或和
P1972 [SDOI2009] HH的项链（https://www.luogu.com.cn/problem/P1972）经典使用树状数组离线查询区间不同数的个数 PointChangeRangeSum OfflineQuery

================================AtCoder================================
D - Islands War（https://atcoder.jp/contests/abc103/tasks/abc103_d）经典贪心加树状数组
F - Absolute Minima （https://atcoder.jp/contests/abc127/tasks/abc127_f）经典离散化与两个树状数组进行加和与计数
Vertex Add Subtree Sum（https://judge.yosupo.jp/problem/vertex_add_subtree_sum）use tree array and dfs order

================================CodeForces================================
F. Range Update Point Query（https://codeforces.com/problemset/problem/1791/F）树状数组维护区间操作数与查询单点值
H2. Maximum Crossings (Hard Version)（https://codeforces.com/contest/1676/problem/H2）树状数组维护前缀区间和
C. Three displays（https://codeforces.com/problemset/problem/987/C）枚举中间数组，使用树状数组维护前后缀最小值
F. Moving Points（https://codeforces.com/contest/1311/problem/F）经典两个离散化树状数组，计数与加和
C. Game on Permutation（https://codeforces.com/contest/1860/problem/C）PointDescendRangeMin
C. Manhattan Subarrays（https://codeforces.com/contest/1550/problem/C）PointAscendPreMax

135. 二维树状数组3（https://loj.ac/p/135）区间修改，区间查询
134. 二维树状数组2（https://loj.ac/p/134）区间修改，单点查询

参考：OI WiKi（https://oi-wiki.org/ds/fenwick/）
"""
from collections import defaultdict, deque
from math import inf
from typing import List

from sortedcontainers import SortedList

from src.data_structure.segment_tree.template import RangeAscendRangeMax
from src.data_structure.sorted_list.template import LocalSortedList
from src.data_structure.tree_array.template import PointAddRangeSum, PointDescendPreMin, RangeAddRangeSum, \
    PointAscendPreMax, PointAscendRangeMax, PointAddRangeSum2D, RangeAddRangeSum2D, PointXorRangeXor, \
    PointDescendRangeMin, PointChangeRangeSum
from src.search.dfs.template import DfsEulerOrder
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1626(scores: List[int], ages: List[int]) -> int:
        # 模板：动态规划与树状数组维护前缀最大值
        n = max(ages)
        tree_array = PointAscendPreMax(n)
        for score, age in sorted(zip(scores, ages)):
            cur = tree_array.pre_max(age) + score
            tree_array.point_ascend(age, cur)
        return tree_array.pre_max(n)

    @staticmethod
    def lc_2193_1(s: str) -> int:
        # 模板：使用树状数组贪心模拟交换构建回文串

        n = len(s)
        lst = list(s)
        ans = 0
        dct = defaultdict(deque)
        for i in range(n):
            dct[lst[i]].append(i)
        tree = PointAddRangeSum(n)
        i, j = 0, n - 1
        while i < j:
            if lst[i] == "":
                i += 1
                continue
            if lst[j] == "":
                j -= 1
                continue
            if lst[i] == lst[j]:
                dct[lst[i]].popleft()
                dct[lst[j]].pop()
                i += 1
                j -= 1
                continue

            if len(dct[lst[j]]) >= 2:
                left = dct[lst[j]][0]
                ans += left - i - tree.range_sum(i + 1, left + 1)
                x = dct[lst[j]].popleft()
                dct[lst[j]].pop()
                lst[x] = ""
                tree.point_add(x + 1, 1)
                j -= 1
            else:
                right = dct[lst[i]][-1]
                ans += j - right - tree.range_sum(right + 1, j + 1)
                x = dct[lst[i]].pop()
                dct[lst[i]].popleft()
                tree.point_add(x + 1, 1)
                lst[x] = ""
                i += 1
        return ans

    @staticmethod
    def lc_2193_2(s: str) -> int:
        # 模板：使用字符串特性贪心模拟交换构建回文串
        n = len(s)
        ans = 0
        for _ in range(n // 2):
            j = s.rindex(s[0])
            if j == 0:
                i = s.index(s[-1])
                ans += i
                s = s[:i] + s[i + 1:-1]
            else:
                ans += len(s) - 1 - j
                s = s[1:j] + s[j + 1:]

        return ans

    @staticmethod
    def lc_2407(nums: List[int], k: int) -> int:
        # 模板：树状数组加线性DP
        n = max(nums)
        ans = 0
        tree = PointAscendRangeMax(n)
        tree.ceil = [0] * (n + 1)
        tree.floor = [0] * (n + 1)
        for num in nums:
            low = num - k
            high = num - 1
            if low < 1:
                low = 1
            if low <= high:
                cur = tree.range_max(low, high)
                cur += 1
            else:
                cur = 1
            if cur > ans:
                ans = cur
            tree.point_ascend(num, cur)
        return ans

    @staticmethod
    def lc_2659(nums: List[int]) -> int:
        # 模板：经典模拟删除，可以使用树状数组也可以使用SortedList也可以使用贪心
        n = len(nums)
        ans = 0
        pre = 1
        dct = {num: i + 1 for i, num in enumerate(nums)}
        tree = PointAddRangeSum(n)
        for num in sorted(nums):
            i = dct[num]
            if pre <= i:
                ans += i - pre + 1 - tree.range_sum(pre, i)
            else:
                ans += n - pre + 1 - tree.range_sum(pre, n) + i - 1 + 1 - tree.range_sum(1, i)
            tree.point_add(i, 1)
            pre = i
        return ans

    @staticmethod
    def lc_2736(nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:

        nodes = set(nums1 + nums2)
        for x, y in queries:
            nodes.add(x)
            nodes.add(y)
        nodes = sorted(nodes)
        dct = {num: i for i, num in enumerate(nodes)}
        k = len(nodes)
        n = len(nums1)
        m = len(queries)
        ans = [-1] * m
        ind = list(range(m))
        ind.sort(key=lambda it: queries[it][0], reverse=True)
        index = list(range(n))
        index.sort(key=lambda it: nums1[it], reverse=True)
        i = 0
        tree = PointAscendPreMax(k, -1)
        for j in ind:
            x, y = queries[j]
            while i < n and nums1[index[i]] >= x:
                value = nums1[index[i]] + nums2[index[i]]
                tree.point_ascend(k - dct[nums2[index[i]]], value)
                i += 1
            ans[j] = tree.pre_max(k - dct[y])
        return ans

    @staticmethod
    def lc_6353(grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        dp = [[inf] * m for _ in range(n)]
        r, c = [PointDescendPreMin(m) for _ in range(n)], [PointDescendPreMin(n) for _ in range(m)]
        dp[n - 1][m - 1] = 1
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if grid[i][j] > 0:
                    dp[i][j] = min(r[i].pre_min(min(j + grid[i][j] + 1, m)),
                                   c[j].pre_min(min(i + grid[i][j] + 1, n))) + 1
                if dp[i][j] <= n * m:
                    r[i].point_descend(j + 1, dp[i][j])
                    c[j].point_descend(i + 1, dp[i][j])
        return -1 if dp[0][0] > n * m else dp[0][0]

    @staticmethod
    def lc_100112_1(nums: List[int]) -> int:
        # 树状数组（单点持续更新为更大值）（区间查询最大值）2380ms
        n = len(nums)
        tmp = [nums[i] - i for i in range(n)]
        ind = sorted(list(set(tmp)))
        dct = {x: i for i, x in enumerate(ind)}
        tree = PointAscendRangeMax(n, -inf)
        for j in range(n):
            num = nums[j]
            i = dct[num - j]
            pre = tree.range_max(1, i + 1) if i + 1 >= 1 else 0
            pre = 0 if pre < 0 else pre
            tree.point_ascend(i + 1, pre + num)
        return tree.range_max(1, n)

    @staticmethod
    def lc_100112_2(nums: List[int]) -> int:
        # 树状数组（单点持续更新为更大值）（前缀区间查询最大值）1748ms
        n = len(nums)
        tmp = [nums[i] - i for i in range(n)]
        ind = sorted(list(set(tmp)))
        dct = {x: i for i, x in enumerate(ind)}
        tree = PointAscendPreMax(n)
        for j in range(n):
            num = nums[j]
            i = dct[num - j]
            pre = tree.pre_max(i + 1)
            pre = 0 if pre < 0 else pre
            tree.point_ascend(i + 1, pre + num)
        return tree.pre_max(n)

    @staticmethod
    def lc_100112_3(nums: List[int]) -> int:
        # 线段树（单点持续更新为更大值）（区间查询最大值）7980ms
        n = len(nums)
        tmp = [nums[i] - i for i in range(n)]
        ind = sorted(list(set(tmp)))
        dct = {x: i for i, x in enumerate(ind)}
        tree = RangeAscendRangeMax(n)
        for j in range(n):
            num = nums[j]
            i = dct[num - j]
            pre = tree.range_max(0, i)
            pre = 0 if pre < 0 else pre
            tree.range_ascend(i, i, pre + num)
        ans = tree.range_max(0, n - 1)
        return ans

    @staticmethod
    def lib_c(ac=FastIO()):
        """template of vertex add subtree sum"""
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        p = ac.read_list_ints()
        for i in range(n - 1):
            dct[p[i]].append(i + 1)
        dfs_euler = DfsEulerOrder(dct)
        tree = PointAddRangeSum(n)
        tree.build([nums[i] for i in dfs_euler.order_to_node])
        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0]:
                u = lst[1]
                x, y = dfs_euler.start[u], dfs_euler.end[u]
                ac.st(tree.range_sum(x + 1, y + 1))
            else:
                u, x = lst[1:]
                ind = dfs_euler.start[u]
                tree.point_add(ind + 1, x)
        return

    @staticmethod
    def lc_100112_1(nums: List[int]) -> int:
        # 树状数组（单点持续更新为更大值）（区间查询最大值）2380ms
        n = len(nums)
        tmp = [nums[i] - i for i in range(n)]
        ind = sorted(list(set(tmp)))
        dct = {x: i for i, x in enumerate(ind)}
        tree = PointAscendRangeMax(n, -inf)
        for j in range(n):
            num = nums[j]
            i = dct[num - j]
            pre = tree.range_max(1, i + 1) if i + 1 >= 1 else 0
            pre = 0 if pre < 0 else pre
            tree.point_ascend(i + 1, pre + num)
        return tree.range_max(1, n)

    @staticmethod
    def lc_100112_2(nums: List[int]) -> int:
        # 树状数组（单点持续更新为更大值）（前缀区间查询最大值）1748ms
        n = len(nums)
        tmp = [nums[i] - i for i in range(n)]
        ind = sorted(list(set(tmp)))
        dct = {x: i for i, x in enumerate(ind)}
        tree = PointAscendPreMax(n)
        for j in range(n):
            num = nums[j]
            i = dct[num - j]
            pre = tree.pre_max(i + 1)
            pre = 0 if pre < 0 else pre
            tree.point_ascend(i + 1, pre + num)
        return tree.pre_max(n)

    @staticmethod
    def lc_100112_3(nums: List[int]) -> int:
        # 线段树（单点持续更新为更大值）（区间查询最大值）7980ms
        n = len(nums)
        tmp = [nums[i] - i for i in range(n)]
        ind = sorted(list(set(tmp)))
        dct = {x: i for i, x in enumerate(ind)}
        tree = RangeAscendRangeMax(n)
        for j in range(n):
            num = nums[j]
            i = dct[num - j]
            pre = tree.range_max(0, i)
            pre = 0 if pre < 0 else pre
            tree.range_ascend(i, i, pre + num)
        ans = tree.range_max(0, n - 1)
        return ans

    @staticmethod
    def lg_5094(ac=FastIO()):

        # 模板：树状数组单点增加值与前缀区间和查询
        n = ac.read_int()
        m = 5 * 10 ** 4

        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda y: y[0])
        tree_sum = PointAddRangeSum(m)
        tree_cnt = PointAddRangeSum(m)
        total_cnt = 0
        total_sum = 0
        ans = 0
        for v, x in nums:
            pre_sum = tree_sum.range_sum(1, x)
            pre_cnt = tree_cnt.range_sum(1, x)
            ans += v * (pre_cnt * x - pre_sum) + v * (total_sum - pre_sum - (total_cnt - pre_cnt) * x)
            tree_sum.point_add(x, x)
            tree_cnt.point_add(x, 1)
            total_cnt += 1
            total_sum += x
        ac.st(ans)
        return

    @staticmethod
    def lg_p2280(ac=FastIO()):
        # 模板：树状数组单点更新区间查询最大值与最小值
        n, q = ac.read_list_ints()
        tree = PointAscendRangeMax(n)
        tree2 = PointDescendRangeMin(n)
        for i in range(n):
            tree.a[i + 1] = ac.read_int()
            tree.point_ascend(i + 1, tree.a[i + 1])
            tree2.a[i + 1] = ac.read_int()
            tree2.point_descend(i + 1, tree.a[i + 1])
        for _ in range(q):
            a, b = ac.read_list_ints()
            ac.st(tree.range_max(a, b) - tree2.range_min(a, b))
        return

    @staticmethod
    def cf_1311f(ac=FastIO()):
        # 模板：经典两个离散化树状数组，计数与加和
        n = ac.read_int()
        ind = list(range(n))
        x = ac.read_list_ints()
        ind.sort(key=lambda it: x[it])
        v = ac.read_list_ints()
        dct = {w: i for i, w in enumerate(sorted(set(v)))}
        m = len(dct)
        tree_cnt = PointAddRangeSum(m)
        tree_tot = PointAddRangeSum(m)
        ans = 0
        for i in ind:
            cur_v = v[i]
            tree_cnt.point_add(dct[cur_v] + 1, 1)
            tree_tot.point_add(dct[cur_v] + 1, x[i])
            pre_cnt = tree_cnt.range_sum(1, dct[cur_v] + 1)
            pre_tot = tree_tot.range_sum(1, dct[cur_v] + 1)
            ans += pre_cnt * x[i] - pre_tot
        ac.st(ans)
        return

    @staticmethod
    def cf_1676h2(ac=FastIO()):
        # 模板：树状数组维护前缀区间和
        for _ in range(ac.read_int()):
            ac.read_int()
            a = ac.read_list_ints()
            ceil = max(a)
            ans = 0
            tree = PointAddRangeSum(ceil)
            x = 0
            for num in a:
                ans += x - tree.range_sum(1, num - 1)
                tree.point_add(num, 1)
                x += 1
            ac.st(ans)
        return

    @staticmethod
    def lg_p1972(ac=FastIO()):
        n = ac.read_int()
        nums = ac.read_list_ints()
        m = ac.read_int()
        queries = [ac.read_list_ints_minus_one() + [i] for i in range(m)]
        ans = [0] * m
        tree = PointAddRangeSum(n)
        queries.sort(key=lambda it: it[1])
        pre = [-1]*(max(nums)+1)
        i = 0
        for ll, rr, ii in queries:
            while i <= rr:
                d = nums[i]
                if pre[d] != -1:
                    tree.point_add(pre[d] + 1, -1)
                pre[d] = i
                tree.point_add(i + 1, 1)
                i += 1

            ans[ii] = tree.range_sum(ll + 1, rr + 1)
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lg_p2068(ac=FastIO()):
        # 模板：树状数组单点更新与区间和查询
        n = ac.read_int()
        w = ac.read_int()
        tree = RangeAddRangeSum(n)
        for _ in range(w):
            lst = ac.read_list_strs()
            a, b = int(lst[1]), int(lst[2])
            if lst[0] == "x":
                tree.range_add(a, a, b)
            else:
                ac.st(tree.range_sum(a, b))
        return

    @staticmethod
    def lg_p1816(ac=FastIO()):

        # 模板：树状数组查询静态区间最小值
        m, n = ac.read_list_ints()
        nums = ac.read_list_ints()
        tree = PointDescendRangeMin(m)
        for i in range(m):
            tree.point_descend(i + 1, nums[i])
        ans = []
        for _ in range(n):
            x, y = ac.read_list_ints()
            ans.append(tree.range_min(x, y))
        ac.lst(ans)
        return

    @staticmethod
    def lg_p3374(ac=FastIO()):
        # 模板：树状数组 单点增减 查询前缀和与区间和
        n, m = ac.read_list_ints()
        tree = PointAddRangeSum(n)
        tree.build(ac.read_list_ints())
        for _ in range(m):
            op, x, y = ac.read_list_ints()
            if op == 1:
                tree.point_add(x, y)
            else:
                ac.st(tree.range_sum(x, y))
        return

    @staticmethod
    def lg_p3368(ac=FastIO()):
        # 模板：树状数组 区间增减 查询前缀和与区间和
        n, m = ac.read_list_ints()
        tree = RangeAddRangeSum(n)
        tree.build(ac.read_list_ints())
        for _ in range(m):
            lst = ac.read_list_ints()
            if len(lst) == 2:
                ac.st(tree.range_sum(lst[1], lst[1]))
            else:
                x, y, k = lst[1:]
                tree.range_add(x, y, k)
        return

    @staticmethod
    def lg_p1908(ac=FastIO()):
        # 模板：树状数组求逆序对
        n = ac.read_int()
        nums = ac.read_list_ints()
        ind = list(range(n))
        ind.sort(key=lambda it: nums[it])
        tree = PointAddRangeSum(n)
        ans = i = cnt = 0
        while i < n:
            val = nums[ind[i]]
            lst = []
            while i < n and nums[ind[i]] == val:
                lst.append(ind[i] + 1)
                ans += cnt - tree.range_sum(1, ind[i] + 1)
                i += 1
            cnt += len(lst)
            for x in lst:
                tree.point_add(x, 1)
        ac.st(ans)
        return

    @staticmethod
    def main(ac=FastIO()):
        # 模板：二维树状数组 区间增减 区间查询
        n, m = ac.read_list_ints()
        tree = RangeAddRangeSum2D(n, m)
        while True:
            lst = ac.read_list_ints()
            if not lst:
                break
            if lst[0] == 1:
                a, b, c, d, x = lst[1:]
                tree.range_add(a, b, c, d, x)
            else:
                a, b, c, d = lst[1:]
                ac.st(tree.range_query(a, b, c, d))
        return

    @staticmethod
    def lg_p1725(ac=FastIO()):
        # 模板：树状数组倒序线性DP，单点更新与区间查询最大值
        n, a, b = ac.read_list_ints()
        n += 1
        nums = ac.read_list_ints()
        tree = PointAscendRangeMax(n + 1)
        tree.point_ascend(n + 1, 0)
        post = 0
        for i in range(n - 1, -1, -1):
            x, y = i + a + 1, i + b + 1
            x = n + 1 if x > n + 1 else x
            y = n + 1 if y > n + 1 else y
            post = tree.range_max(x, y)
            tree.point_ascend(i + 1, post + nums[i])
        ac.st(post + nums[0])
        return

    @staticmethod
    def lg_p3586(ac=FastIO()):
        # 模板：离线查询、离散化树状数组，单点增减，前缀和查询
        n, m = ac.read_list_ints()
        value = {0}
        lst = []
        for _ in range(m):
            cur = ac.read_list_strs()
            if cur[0] == "U":
                k, a = [int(w) for w in cur[1:]]
                value.add(a)
                lst.append([1, k, a])
            else:
                c, s = [int(w) for w in cur[1:]]
                value.add(s)
                lst.append([2, c, s])
        value = sorted(list(value))
        ind = {num: i for i, num in enumerate(value)}
        length = len(ind)

        tree_cnt = PointAddRangeSum(length)
        tree_sum = PointAddRangeSum(length)
        nums = [0] * n
        total_s = 0
        total_c = 0

        for op, a, b in lst:
            if op == 1:
                if nums[a - 1]:
                    tree_cnt.point_add(ind[nums[a - 1]], -1)
                    tree_sum.point_add(ind[nums[a - 1]], -nums[a - 1])
                    total_s -= nums[a - 1]
                    total_c -= 1
                nums[a - 1] = b
                if nums[a - 1]:
                    tree_cnt.range_sum(ind[nums[a - 1]], 1)
                    tree_sum.range_sum(ind[nums[a - 1]], nums[a - 1])
                    total_s += nums[a - 1]
                    total_c += 1
            else:
                c, s = a, b
                less_than_s = tree_cnt.range_sum(1, ind[s] - 1)
                less_than_s_sum = tree_sum.range_sum(1, ind[s] - 1)
                if (total_c - less_than_s) * s + less_than_s_sum >= c * s:
                    ac.st("TAK")
                else:
                    ac.st("NIE")
        return

    @staticmethod
    def lg_p1198(ac=FastIO()):
        # 模板：树状数组查询区间最大值
        m, d = ac.read_list_ints()
        t = 0
        tree = PointAscendRangeMax(m + 1)
        i = 1
        for _ in range(m):
            op, x = ac.read_list_strs()
            if op == "A":
                x = (int(x) + t) % d
                tree.point_ascend(i, x)
                i += 1
            else:
                x = int(x)
                t = tree.range_max(i - x, i - 1)
                ac.st(t)
        return

    @staticmethod
    def lg_p4868(ac=FastIO()):
        # 模板：经典转换公式，使用两个树状数组维护前缀和的前缀和

        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        tree1 = PointAddRangeSum(n)
        tree1.build(nums)
        tree2 = PointAddRangeSum(n)
        tree2.build([nums[i] * (i + 1) for i in range(n)])
        for _ in range(m):
            lst = ac.read_list_strs()
            if lst[0] == "Modify":
                i, x = [int(w) for w in lst[1:]]
                y = nums[i - 1]
                nums[i - 1] = x
                tree1.point_add(i, x - y)
                tree2.point_add(i, i * (x - y))
            else:
                i = int(lst[1])
                ac.st((i + 1) * tree1.range_sum(1, i) - tree2.range_sum(1, i))
        return

    @staticmethod
    def lg_p5463(ac=FastIO()):
        # 模板：经典使用树状数组维护前缀计数，枚举最大值计算所有区间数贡献
        n = ac.read_int()
        nums = ac.read_list_ints()
        lst = sorted(list(set(nums)))
        ind = {num: i + 1 for i, num in enumerate(lst)}
        m = len(ind)
        tree = PointAddRangeSum(m)
        ans = 0
        for i in range(n - 1, -1, -1):
            left = i + 1
            right = tree.range_sum(1, ind[nums[i]] - 1)
            ans += left * right
            # 取 nums[i] 作为区间的数又 n-i 个右端点取法
            tree.point_add(ind[nums[i]], n - i)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6225(ac=FastIO()):
        # 模板：经典使用树状数组维护前缀异或和
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()

        tree_odd = PointXorRangeXor(n)
        tree_even = PointXorRangeXor(n)
        for i in range(n):
            # 也可以使用对应子数组进行初始化
            if i % 2 == 0:
                tree_odd.point_xor(i + 1, nums[i])
            else:
                tree_even.point_xor(i + 1, nums[i])

        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                i, x = lst[1:]
                a = nums[i - 1]
                if i % 2 == 0:
                    tree_even.point_xor(i, a ^ x)
                else:
                    tree_odd.point_xor(i, a ^ x)
                nums[i - 1] = x
            else:
                left, right = lst[1:]
                if (right - left + 1) % 2 == 0:
                    ac.st(0)
                else:
                    # 如果是奇数长度则为 left 开始每隔 2 的元素异或和
                    if left % 2:
                        ac.st(tree_odd.range_xor(left, right))
                    else:
                        ac.st(tree_even.range_xor(left, right))
        return

    @staticmethod
    def abc_127f(ac=FastIO()):
        # 模板：经典离散化与两个树状数组进行加和与计数
        queries = [ac.read_list_ints() for _ in range(ac.read_int())]
        nodes = set()
        for lst in queries:
            if len(lst) > 1:
                a, _ = lst[1:]
                nodes.add(a)
        nodes = sorted(nodes)
        ind = {num: i for i, num in enumerate(nodes)}
        n = len(ind)
        ans = 0
        tree_sum = PointAddRangeSum(n)
        tree_cnt = PointAddRangeSum(n)
        pre = LocalSortedList()
        for lst in queries:
            if lst[0] == 1:
                a, b = lst[1:]
                ans += b
                tree_sum.point_add(ind[a] + 1, a)
                tree_cnt.point_add(ind[a] + 1, 1)
                pre.add(a)
            else:
                m = len(pre)
                if m % 2 == 0:
                    i = m // 2 - 1
                else:
                    i = m // 2
                val = pre[i]
                i = ind[val]
                left = val * tree_cnt.range_sum(1, i + 1) - tree_sum.range_sum(1, i + 1)
                if i + 2 <= n:
                    right = -val * tree_cnt.range_sum(i + 2, n) + tree_sum.range_sum(i + 2, n)
                else:
                    right = 0
                ac.lst([val, left + right + ans])
        return

    @staticmethod
    def cf_987c(ac=FastIO()):
        # 模板：枚举中间数组，使用树状数组维护前后缀最小值
        n = ac.read_int()
        s = ac.read_list_ints()
        c = ac.read_list_ints()

        nodes = sorted(list(set(s)) + [0, 10 ** 9 + 1])
        dct = {num: i + 1 for i, num in enumerate(nodes)}
        m = len(nodes)

        pre = [inf] * n
        tree = PointDescendPreMin(m)
        for i in range(n):
            pre[i] = tree.pre_min(dct[s[i]] - 1)
            tree.point_descend(dct[s[i]], c[i])

        post = [inf] * n
        tree = PointDescendPreMin(m)
        for i in range(n - 1, -1, -1):
            post[i] = tree.pre_min(m - dct[s[i]])
            tree.point_descend(m - dct[s[i]] + 1, c[i])

        ans = inf
        if n >= 3:
            ans = min(pre[i] + post[i] + c[i] for i in range(1, n - 1))
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def cf_1860c(ac=FastIO()):
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints_minus_one()
            tree = PointDescendRangeMin(n, 2)

            for i in range(n):
                x = nums[i]
                if x == 0:
                    cur = 1
                else:
                    low = tree.range_min(1, x)
                    if low == 0:
                        cur = 1
                    elif low == 1:
                        cur = 0
                    else:
                        cur = 1
                tree.point_descend(x + 1, cur)
            ans = [tree.range_min(x, x) for x in range(1, n + 1)]
            ac.st(n - sum(ans))
        return

    @staticmethod
    def lc_1505_1(num: str, k: int) -> str:

        # 模板：经典使用树状数组模拟
        n = len(num)
        dct = defaultdict(deque)
        for i, d in enumerate(num):
            dct[d].append(i)
        # 使用树状数组模拟交换过程
        tree = PointAddRangeSum(n)
        ans = ""
        for i in range(n):
            # 添加第 i 个数字
            cur = i
            for d in range(10):
                # 找还有的数字
                if dct[str(d)]:
                    i = dct[str(d)][0]
                    ind = i + tree.range_sum(i + 1, n)
                    # 索引加上移动之后的位置与第i个相隔距离在代价承受范围内
                    if ind - cur <= k:
                        ans += str(d)
                        k -= ind - cur
                        tree.point_add(i + 1, 1)
                        dct[str(d)].popleft()
                        break
        return ans

    @staticmethod
    def lc_1505_2(num: str, k: int) -> str:
        ind = [deque() for _ in range(10)]

        # 按照数字存好索引
        n = len(num)
        for i in range(n):
            ind[int(num[i])].append(i)

        move = SortedList()
        ans = ""
        for i in range(n):
            # 添加第i个数字
            for x in range(10):
                if ind[x]:
                    # 找还有的数字
                    j = ind[x][0]
                    dis = len(move) - move.bisect_right(j)
                    # 索引加上移动之后的位置与第i个相隔距离在代价承受范围内
                    if dis + j - i <= k:
                        move.add(ind[x].popleft())
                        ans += str(x)
                        k -= dis + j - i
                        break
        return ans


class LC307:

    def __init__(self, nums: List[int]):
        n = len(nums)
        self.tree = PointChangeRangeSum(n)
        self.tree.build(nums)

    def update(self, index: int, val: int) -> None:
        self.tree.point_change(index + 1, val)

    def sum_range(self, left: int, right: int) -> int:
        return self.tree.range_sum(left + 1, right + 1)


class LC308:
    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        self.matrix = matrix
        self.tree = PointAddRangeSum2D(m, n)
        for i in range(m):
            for j in range(n):
                self.tree.point_add(i + 1, j + 1, matrix[i][j])

    def update(self, row: int, col: int, val: int) -> None:
        # 注意这里是修改为 val
        self.tree.point_add(row + 1, col + 1, val - self.matrix[row][col])
        self.matrix[row][col] = val

    def sum_region(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.tree.range_sum(row1 + 1, col1 + 1, row2 + 1, col2 + 1)
