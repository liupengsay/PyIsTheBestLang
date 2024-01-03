"""
Algorithm：tree_array|tree_matrix
Description：range_add|range_sum

====================================LeetCode====================================
307（https://leetcode.cn/problems/range-sum-query-mutable）PointChangeRangeSum
1409（https://leetcode.cn/problems/queries-on-a-permutation-with-key/）tree_array|implemention
1626（https://leetcode.cn/problems/best-team-with-no-conflicts/）tree_array|prefix_maximum|dp
2617（https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/）tree_array|prefix_minimum
308（https://leetcode.cn/problems/range-sum-query-2d-mutable/）tree_matrix|RangeAddRangeSum
2659（https://leetcode.cn/problems/make-array-empty/submissions/）implemention|tree_array|sortedList|greedy
1505（https://leetcode.cn/problems/minimum-possible-integer-after-at-most-k-adjacent-swaps-on-digits/）tree_array|implemention|counter|sorted_list
2193（https://leetcode.cn/problems/minimum-number-of-moves-to-make-palindrome/description/）tree_array|greedy|implemention|P5041
2407（https://leetcode.cn/problems/longest-increasing-subsequence-ii/description/）tree_array|liner_dp
2926（https://leetcode.cn/problems/maximum-balanced-subsequence-sum/）discretization|tree_array|liner_dp
2736（https://leetcode.cn/problems/maximum-sum-queries/）PointAddPreMax

=====================================LuoGu======================================
P2068（https://www.luogu.com.cn/problem/P2068）PointAddRangeSum
P2345（https://www.luogu.com.cn/problem/P2345）tree_array|counter|range_sum
P2357（https://www.luogu.com.cn/problem/P2357）tree_array|range_sum
P2781（https://www.luogu.com.cn/problem/P2781）tree_array|range_sum
P5200（https://www.luogu.com.cn/problem/P5200）tree_array|greedy|implemention
P3374（https://www.luogu.com.cn/problem/P3374）tree_array|RangeAddRangeSum
P3368（https://www.luogu.com.cn/problem/P3368）tree_array|RangeAddRangeSum
P5677（https://www.luogu.com.cn/problem/P5677）tree_array|RangeAddRangeSum
P5094（https://www.luogu.com.cn/problem/P5094）tree_array|RangeAddRangeSum
P1816（https://www.luogu.com.cn/problem/P1816）tree_array|range_min
P1725（https://www.luogu.com.cn/problem/P1725）reverse_order|liner_dp|PointAscendRangeMax
P3586（https://www.luogu.com.cn/problem/P3586）offline_query|discretization|tree_array|PointAddPreSum
P1198（https://www.luogu.com.cn/problem/P1198）tree_array|range_max
P4868（https://www.luogu.com.cn/problem/P4868）math|tree_array|prefix_sum_of_prefix_sum
P5463（https://www.luogu.com.cn/problem/P5463）tree_array|counter|brute_force|contribution_method
P6225（https://www.luogu.com.cn/problem/P6225）tree_array|prefix_xor
P1972（https://www.luogu.com.cn/problem/P1972）tree_array|offline_query|range_unique|PointChangeRangeSum

====================================AtCoder=====================================
ABC103D（https://atcoder.jp/contests/abc103/tasks/abc103_d）greedy|tree_array
ABC127F（https://atcoder.jp/contests/abc127/tasks/abc127_f）discretization|tree_array|counter


===================================CodeForces===================================
1791F（https://codeforces.com/problemset/problem/1791/F）tree_array
1676H2（https://codeforces.com/contest/1676/problem/H2）tree_array|pre_sum
987C（https://codeforces.com/problemset/problem/987/C）brute_force|tree_array|prefix_suffix|pre_min
1311F（https://codeforces.com/contest/1311/problem/F）discretization|tree_array|counter
1860C（https://codeforces.com/contest/1860/problem/C）PointDescendRangeMin
1550C（https://codeforces.com/contest/1550/problem/C）PointAscendPreMax
1679C（https://codeforces.com/contest/1679/problem/C）PointAddRangeSum

1（https://judge.yosupo.jp/problem/vertex_add_subtree_sum）tree_array|dfs_order
135. tree_matrix|3（https://loj.ac/p/135）range_change|range_sum
134. tree_matrix|2（https://loj.ac/p/134）range_change|range_sum

"""
from collections import defaultdict, deque
from typing import List

from src.data_structure.segment_tree.template import RangeAscendRangeMax
from src.data_structure.sorted_list.template import SortedList
from src.data_structure.tree_array.template import PointAddRangeSum, PointDescendPreMin, RangeAddRangeSum, \
    PointAscendPreMax, PointAscendRangeMax, PointAddRangeSum2D, RangeAddRangeSum2D, PointXorRangeXor, \
    PointDescendRangeMin, PointChangeRangeSum
from src.search.dfs.template import DfsEulerOrder
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1626(scores: List[int], ages: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/best-team-with-no-conflicts/
        tag: tree_array|prefix_maximum|dp
        """
        n = max(ages)
        tree_array = PointAscendPreMax(n)
        for score, age in sorted(zip(scores, ages)):
            cur = tree_array.pre_max(age) + score
            tree_array.point_ascend(age, cur)
        return tree_array.pre_max(n)

    @staticmethod
    def lc_2193_1(s: str) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-moves-to-make-palindrome/description/
        tag: tree_array|greedy|implemention|P5041
        """
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
        """
        url: https://leetcode.cn/problems/minimum-number-of-moves-to-make-palindrome/description/
        tag: tree_array|greedy|implemention|P5041
        """
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
        """
        url: https://leetcode.cn/problems/longest-increasing-subsequence-ii/description/
        tag: tree_array|liner_dp
        """
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
        """
        url: https://leetcode.cn/problems/make-array-empty/submissions/
        tag: implemention|tree_array|sortedList|greedy
        """
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
        """
        url: https://leetcode.cn/problems/maximum-sum-queries/
        tag: PointAddPreMax
        """
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
    def lc_2617(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/
        tag: tree_array|prefix_minimum
        """
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
    def lc_2926_1(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-balanced-subsequence-sum/
        tag: discretization|tree_array|liner_dp
        """
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
    def lc_2926_2(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-balanced-subsequence-sum/
        tag: discretization|tree_array|liner_dp
        """
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
    def lc_2926_3(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-balanced-subsequence-sum/
        tag: discretization|tree_array|liner_dp
        """
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
    def library_checker_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/vertex_add_subtree_sum
        tag: tree_array|dfs_order|classical|hard
        """
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
    def lg_p5094(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5094
        tag: tree_array|RangeAddRangeSumtree_array|RangeAddRangeSum
        """
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
    def lg_xxxx(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/xxxx
        tag:
        """
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
        """
        url: https://codeforces.com/contest/1311/problem/F
        tag: discretization|tree_array|counter
        """
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
        """
        url: https://codeforces.com/contest/1676/problem/H2
        tag: tree_array|pre_sum
        """
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
        """
        url: https://www.luogu.com.cn/problem/P1972
        tag: tree_array|offline_query|range_unique|PointChangeRangeSum
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        m = ac.read_int()
        queries = [ac.read_list_ints_minus_one() + [i] for i in range(m)]
        ans = [0] * m
        tree = PointAddRangeSum(n)
        queries.sort(key=lambda it: it[1])
        pre = [-1] * (max(nums) + 1)
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
        """
        url: https://www.luogu.com.cn/problem/P2068
        tag: PointAddRangeSum
        """
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
        """
        url: https://www.luogu.com.cn/problem/P1816
        tag: tree_array|range_min
        """
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
        """
        url: https://www.luogu.com.cn/problem/P3374
        tag: tree_array|RangeAddRangeSum
        """
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
        """
        url: https://www.luogu.com.cn/problem/P3368
        tag: tree_array|RangeAddRangeSum
        """
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
    def main(ac=FastIO()):
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
        """
        url: https://www.luogu.com.cn/problem/P1725
        tag: reverse_order|liner_dp|PointAscendRangeMax
        """
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
        """
        url: https://www.luogu.com.cn/problem/P3586
        tag: offline_query|discretization|tree_array|PointAddPreSum
        """
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
        """
        url: https://www.luogu.com.cn/problem/P1198
        tag: tree_array|range_max
        """
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
        """
        url: https://www.luogu.com.cn/problem/P4868
        tag: math|tree_array|prefix_sum_of_prefix_sum
        """
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
        """
        url: https://www.luogu.com.cn/problem/P5463
        tag: tree_array|counter|brute_force|contribution_method
        """
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
            tree.point_add(ind[nums[i]], n - i)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6225(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6225
        tag: tree_array|prefix_xor
        """
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()

        tree_odd = PointXorRangeXor(n)
        tree_even = PointXorRangeXor(n)
        for i in range(n):
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
                    if left % 2:
                        ac.st(tree_odd.range_xor(left, right))
                    else:
                        ac.st(tree_even.range_xor(left, right))
        return

    @staticmethod
    def abc_127f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc127/tasks/abc127_f
        tag: discretization|tree_array|counter
        """
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
        pre = SortedList()
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
        """
        url: https://codeforces.com/problemset/problem/987/C
        tag: brute_force|tree_array|prefix_suffix|pre_min
        """
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
    def cf_1679c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1679/problem/C
        tag: PointAddRangeSum
        """
        n, q = ac.read_list_ints()
        row = [0] * n
        col = [0] * n
        row_tree = PointAddRangeSum(n)
        col_tree = PointAddRangeSum(n)
        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y = [w - 1 for w in lst[1:]]
                row[x] += 1
                col[y] += 1
                if row[x] == 1:
                    row_tree.point_add(x + 1, 1)
                if col[y] == 1:
                    col_tree.point_add(y + 1, 1)
            elif lst[0] == 2:
                x, y = [w - 1 for w in lst[1:]]
                row[x] -= 1
                col[y] -= 1
                if row[x] == 0:
                    row_tree.point_add(x + 1, -1)
                if col[y] == 0:
                    col_tree.point_add(y + 1, -1)
            else:
                x1, y1, x2, y2 = [w - 1 for w in lst[1:]]
                if (row_tree.range_sum(x1 + 1, x2 + 1) == x2 - x1 + 1
                        or col_tree.range_sum(y1 + 1, y2 + 1) == y2 - y1 + 1):
                    ac.st("Yes")
                    continue
                ac.st("No")
        return

    @staticmethod
    def cf_1860c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1860/problem/C
        tag: PointDescendRangeMin
        """
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
        """
        url: https://leetcode.cn/problems/minimum-possible-integer-after-at-most-k-adjacent-swaps-on-digits/
        tag: tree_array|implemention|counter|sorted_list
        """
        n = len(num)
        dct = defaultdict(deque)
        for i, d in enumerate(num):
            dct[d].append(i)
        tree = PointAddRangeSum(n)
        ans = ""
        for i in range(n):
            cur = i
            for d in range(10):
                if dct[str(d)]:
                    i = dct[str(d)][0]
                    ind = i + tree.range_sum(i + 1, n)
                    if ind - cur <= k:
                        ans += str(d)
                        k -= ind - cur
                        tree.point_add(i + 1, 1)
                        dct[str(d)].popleft()
                        break
        return ans

    @staticmethod
    def lc_1505_2(num: str, k: int) -> str:
        """
        url: https://leetcode.cn/problems/minimum-possible-integer-after-at-most-k-adjacent-swaps-on-digits/
        tag: tree_array|implemention|counter|sorted_list
        """
        ind = [deque() for _ in range(10)]
        n = len(num)
        for i in range(n):
            ind[int(num[i])].append(i)

        move = SortedList()
        ans = ""
        for i in range(n):
            for x in range(10):
                if ind[x]:
                    j = ind[x][0]
                    dis = len(move) - move.bisect_right(j)
                    if dis + j - i <= k:
                        move.add(ind[x].popleft())
                        ans += str(x)
                        k -= dis + j - i
                        break
        return ans

    @staticmethod
    def lc_307():
        """
        url: https://leetcode.com/problems/range-sum-query-mutable
        tag: PointChangeRangeSum
        """
        class NumArray:

            def __init__(self, nums: List[int]):
                n = len(nums)
                self.tree = PointChangeRangeSum(n)
                self.tree.build(nums)

            def update(self, index: int, val: int) -> None:
                self.tree.point_change(index + 1, val)

            def sum_range(self, left: int, right: int) -> int:
                return self.tree.range_sum(left + 1, right + 1)

        return NumArray

    @staticmethod
    def lc_308():
        """
        url: https://leetcode.com/problems/range-sum-query-2d-mutable/
        tag: tree_matrix|RangeAddRangeSum
        """
        class NumMatrix:
            def __init__(self, matrix: List[List[int]]):
                m, n = len(matrix), len(matrix[0])
                self.matrix = matrix
                self.tree = PointAddRangeSum2D(m, n)
                for i in range(m):
                    for j in range(n):
                        self.tree.point_add(i + 1, j + 1, matrix[i][j])

            def update(self, row: int, col: int, val: int) -> None:
                self.tree.point_add(row + 1, col + 1, val - self.matrix[row][col])
                self.matrix[row][col] = val

            def sum_region(self, row1: int, col1: int, row2: int, col2: int) -> int:
                return self.tree.range_sum(row1 + 1, col1 + 1, row2 + 1, col2 + 1)

        return NumMatrix
