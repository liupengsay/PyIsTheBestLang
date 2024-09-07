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
2916（https://leetcode.cn/problems/subarrays-distinct-element-sum-of-squares-ii/）range_add|range_sum|contribution_method|linear_dp
100317（https://leetcode.cn/problems/peaks-in-array/）tree_array|implemention

=====================================LuoGu======================================
P2068（https://www.luogu.com.cn/problem/P2068）PointAddRangeSum
P2345（https://www.luogu.com.cn/problem/P2345）tree_array|counter|range_sum
P2357（https://www.luogu.com.cn/problem/P2357）tree_array|range_sum
P2781（https://www.luogu.com.cn/problem/P2781）tree_array|range_sum
P5200（https://www.luogu.com.cn/problem/P5200）tree_array|greedy|implemention
P3374（https://www.luogu.com.cn/problem/P3374）tree_array|RangeAddRangeSum
P3368（https://www.luogu.com.cn/problem/P3368）tree_array|RangeAddRangeSum
P5094（https://www.luogu.com.cn/problem/P5094）tree_array|RangeAddRangeSum
P1816（https://www.luogu.com.cn/problem/P1816）tree_array|range_min
P1725（https://www.luogu.com.cn/problem/P1725）reverse_order|liner_dp|PointAscendRangeMax
P3586（https://www.luogu.com.cn/problem/P3586）offline_query|discretization|tree_array|PointAddPreSum
P1198（https://www.luogu.com.cn/problem/P1198）tree_array|range_max
P4868（https://www.luogu.com.cn/problem/P4868）math|tree_array|prefix_sum_of_prefix_sum
P5463（https://www.luogu.com.cn/problem/P5463）tree_array|counter|brute_force|contribution_method
P6225（https://www.luogu.com.cn/problem/P6225）tree_array|prefix_xor
P1972（https://www.luogu.com.cn/problem/P1972）tree_array|offline_query|range_unique|PointChangeRangeSum
P5041（https://www.luogu.com.cn/problem/P5041）tree_array|implemention|classical

====================================AtCoder=====================================
ABC103D（https://atcoder.jp/contests/abc103/tasks/abc103_d）greedy|tree_array
ABC127F（https://atcoder.jp/contests/abc127/tasks/abc127_f）discretization|tree_array|counter
ABC287G（https://atcoder.jp/contests/abc287/tasks/abc287_g）segment_tree|range_sum|dynamic|offline|tree_array|bisect_right
ABC306F（https://atcoder.jp/contests/abc306/tasks/abc306_f）tree_array|contribution_method|classical
ABC286F（https://atcoder.jp/contests/abc283/tasks/abc283_f）point_descend|pre_min|tree_array|classical
ABC276F（https://atcoder.jp/contests/abc276/tasks/abc276_f）expectation|comb|tree_array|contribution_method|classical
ABC256F（https://atcoder.jp/contests/abc256/tasks/abc256_f）tree_array|cumulative_cumulative_cumulative_sum|math|classical
ABC250E（https://atcoder.jp/contests/abc250/tasks/abc250_e）tree_array|point_ascend|pre_max|implemention|set|classical
ABC231F（https://atcoder.jp/contests/abc231/tasks/abc231_f）discretize|tree_array|inclusion_exclusion|two_pointer
ABC351F（https://atcoder.jp/contests/abc351/tasks/abc351_f）tree_array|discretize|classical
ABC221E（https://atcoder.jp/contests/abc221/tasks/abc221_e）tree_array|contribution_method
ABC353G（https://atcoder.jp/contests/abc353/tasks/abc353_g）point_ascend|range_max|pre_max|classical
ABC356F（https://atcoder.jp/contests/abc356/tasks/abc356_f）tree_array|binary_search|bisect_right|classical
ABC368G（https://atcoder.jp/contests/abc368/tasks/abc368_g）point_add|range_sum|observation|data_range
ABC369F（https://atcoder.jp/contests/abc369/tasks/abc369_f）tree_array|point_ascend|pre_max_index|construction|specific_plan

===================================CodeForces===================================
1791F（https://codeforces.com/problemset/problem/1791/F）tree_array|data_range|union_find_right|limited_operation
1676H2（https://codeforces.com/contest/1676/problem/H2）tree_array|pre_sum
987C（https://codeforces.com/problemset/problem/987/C）brute_force|tree_array|prefix_suffix|pre_min
1311F（https://codeforces.com/contest/1311/problem/F）discretization|tree_array|counter
1860C（https://codeforces.com/contest/1860/problem/C）PointDescendRangeMin
1550C（https://codeforces.com/contest/1550/problem/C）PointAscendPreMax
1679C（https://codeforces.com/contest/1679/problem/C）PointAddRangeSum
1722E（https://codeforces.com/problemset/problem/1722/E）data_range|matrix_prefix_sum|classical|can_be_discretization_hard_version|tree_array_2d
1430E（https://codeforces.com/problemset/problem/1430/E）tree_array|classical|implemention|point_add|range_sum|pre_sum
1788E（https://codeforces.com/problemset/problem/1788/E）linear_dp|tree_array|point_ascend|pre_max
677D（https://codeforces.com/problemset/problem/677/D）layered_bfs|tree_array|two_pointer|partial_order|implemention|classical
1667B（https://codeforces.com/problemset/problem/1667/B）tree_array|classical|prefix_sum

=====================================LibraryChecker=====================================
1（https://judge.yosupo.jp/problem/vertex_add_subtree_sum）tree_array|dfs_order
135. tree_matrix|3（https://loj.ac/p/135）range_change|range_sum
134. tree_matrix|2（https://loj.ac/p/134）range_change|range_sum
4（https://codeforces.com/edu/course/2/lesson/4/3/practice/contest/274545/problem/A）tree_array|point_set|range_sum|inversion

"""
from collections import defaultdict, deque
from typing import List

from src.data_structure.segment_tree.template import RangeAscendRangeMax
from src.data_structure.sorted_list.template import SortedList
from src.data_structure.tree_array.template import PointAddRangeSum, PointDescendPreMin, RangeAddRangeSum, \
    PointAscendPreMax, PointAscendRangeMax, PointAddRangeSum2D, RangeAddRangeSum2D, PointXorRangeXor, \
    PointDescendRangeMin, PointChangeRangeSum, PointDescendPostMin, PointAscendPreMaxIndex
from src.mathmatics.comb_perm.template import Combinatorics
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
        tag: tree_array|RangeAddRangeSum
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
                    tree_cnt.point_add(ind[nums[a - 1]], 1)
                    tree_sum.point_add(ind[nums[a - 1]], nums[a - 1])
                    total_s += nums[a - 1]
                    total_c += 1
            else:
                c, s = a, b
                less_than_s = tree_cnt.range_sum(0, ind[s])
                less_than_s_sum = tree_sum.range_sum(0, ind[s])
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
        n, q = ac.read_list_ints()  # TLE
        nums = ac.read_list_ints()

        tree_odd = PointXorRangeXor(n)
        tree_even = PointXorRangeXor(n)
        for i in range(n):
            if i % 2:
                tree_odd.point_xor(i, nums[i])
            else:
                tree_even.point_xor(i, nums[i])

        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                i, x = lst[1:]
                i -= 1
                if i % 2:
                    tree_odd.point_xor(i, nums[i] ^ x)
                else:
                    tree_even.point_xor(i, nums[i] ^ x)
                nums[i] = x
            else:
                left, right = lst[1:]
                left -= 1
                right -= 1
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
                    ac.yes()
                    continue
                ac.no()
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

    @staticmethod
    def library_check_4(ac=FastIO()):
        """
        url: https://codeforces.com/edu/course/2/lesson/4/3/practice/contest/274545/problem/A
        tag: segment_tree|point_set|range_sum|inversion
        """
        n = ac.read_int()
        tree = PointAddRangeSum(n, 0)
        nums = ac.read_list_ints()
        ans = [0] * n
        for j in range(n):
            i = n + 1 - nums[j]
            ans[j] = tree.range_sum(1, i)
            tree.point_add(i, 1)
        ac.lst(ans)
        return

    @staticmethod
    def abc_287g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc287/tasks/abc287_g
        tag: segment_tree|range_sum|dynamic|offline|tree_array|bisect_right
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        q = ac.read_int()
        queries = [ac.read_list_ints() for _ in range(q)]
        nodes = set()
        for a, _ in nums:
            nodes.add(a)
        for lst in queries:
            if lst[0] == 1:
                nodes.add(lst[2])
        nodes = sorted(nodes)
        ind = {num: i + 1 for i, num in enumerate(nodes)}
        n = len(nodes)
        tree1 = PointAddRangeSum(n)
        tree2 = PointAddRangeSum(n)
        tot1 = tot2 = 0
        for a, b in nums:
            tree1.point_add(ind[a], b)
            tree2.point_add(ind[a], a * b)
            tot1 += b
            tot2 += a * b

        for lst in queries:
            if lst[0] < 3:
                x, y = lst[1:]
                x -= 1
                a, b = nums[x]
                tree1.point_add(ind[a], -b)
                tree2.point_add(ind[a], -a * b)
                tot1 -= b
                tot2 -= a * b
                if lst[0] == 1:
                    nums[x][0] = y
                else:
                    nums[x][1] = y
                a, b = nums[x]
                tree1.point_add(ind[a], b)
                tree2.point_add(ind[a], a * b)
                tot1 += b
                tot2 += a * b
            else:
                x = lst[1]
                if tot1 < x:
                    ac.st(-1)
                    continue
                i = tree1.bisect_right(tot1 - x)
                ans = tree2.range_sum(1, i) if i else 0
                rest = tot1 - x - tree1.range_sum(1, i) if i else tot1 - x
                ans += rest * nodes[i]
                ac.st(tot2 - ans)
        return

    @staticmethod
    def lc_2916(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/subarrays-distinct-element-sum-of-squares-ii/
        tag: range_add|range_sum|contribution_method|linear_dp

        """
        n = len(nums)
        mod = 10 ** 9 + 7
        ans = dp = 0
        dct = dict()
        tree = RangeAddRangeSum(n)
        for i in range(n):
            num = nums[i]
            if num not in dct:
                dp += 2 * tree.range_sum(1, i + 1) + i + 1
                tree.range_add(1, i + 1, 1)

            else:
                j = dct[num]
                dp += 2 * tree.range_sum(j + 2, i + 1) + i - j
                tree.range_add(j + 2, i + 1, 1)
            ans += dp
            dct[num] = i
            ans %= mod
            dp %= mod
        return ans

    @staticmethod
    def cf_1722e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1722/E
        tag: data_range|matrix_prefix_sum|classical|can_be_discretization_hard_version|tree_array_2d
        """

        for _ in range(ac.read_int()):
            k, q = ac.read_list_ints()
            rec = [ac.read_list_ints() for _ in range(k)]
            qur = [ac.read_list_ints() for _ in range(q)]
            n = max([y for _, y in rec] + [ls[3] for ls in qur])
            m = max([y for y, _ in rec] + [ls[2] for ls in qur])
            tree_2d = PointAddRangeSum2D(m, n)
            for x, y in rec:
                tree_2d.point_add(x, y, x * y)

            for hs, ws, hb, wb in qur:
                ans = tree_2d.range_sum(hs + 1, ws + 1, hb - 1, wb - 1)
                ac.st(ans)
        return

    @staticmethod
    def abc_306f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc306/tasks/abc306_f
        tag: tree_array|contribution_method|classical
        """
        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        nodes = []
        for num in nums:
            nodes.extend(num)
        nodes.sort()
        ind = {num: i + 1 for i, num in enumerate(nodes)}

        nums = [[ind[x] for x in ls] for ls in nums]
        k = len(ind)
        tree = PointAddRangeSum(k)
        for x in nums[-1]:
            tree.point_add(x, 1)
        ans = 0
        for i in range(n - 2, -1, -1):
            for num in nums[i]:
                if num > 1:
                    ans += tree.range_sum(1, num - 1) + (n - i - 1)
                else:
                    ans += (n - i - 1)
            ans += (n - 1 - i) * (1 + m - 1) * (m - 1) // 2
            for x in nums[i]:
                tree.point_add(x, 1)
        ac.st(ans)
        return

    @staticmethod
    def abc_283f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc283/tasks/abc283_f
        tag: manhattan_distance|point_descend|pre_min|tree_array|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()

        ans = [inf] * n
        tree1 = PointDescendPreMin(n)
        tree2 = PointDescendPreMin(n)
        for i in range(n):
            if nums[i] > 1:
                pre = tree1.pre_min(nums[i] - 1)
                ans[i] = min(ans[i], nums[i] + i + pre)
            if nums[i] < n:
                pre = tree2.pre_min(n - nums[i])
                ans[i] = min(ans[i], -nums[i] + i + pre)

            tree1.point_descend(nums[i], -nums[i] - i)
            tree2.point_descend(n + 1 - nums[i], nums[i] - i)

        tree1 = PointDescendPreMin(n)
        tree2 = PointDescendPreMin(n)
        for i in range(n - 1, -1, -1):
            if nums[i] > 1:
                pre = tree1.pre_min(nums[i] - 1)
                ans[i] = min(ans[i], nums[i] - i + pre)
            if nums[i] < n:
                pre = tree2.pre_min(n - nums[i])
                ans[i] = min(ans[i], -nums[i] - i + pre)

            tree1.point_descend(nums[i], -nums[i] + i)
            tree2.point_descend(n + 1 - nums[i], nums[i] + i)
        ac.lst(ans)
        return

    @staticmethod
    def abc_276f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc276/tasks/abc276_f
        tag: expectation|comb|tree_array|contribution_method|classical
        """
        mod = 998244353
        ac.read_int()
        nums = ac.read_list_ints()
        tot = 0
        m = 2 * 10 ** 5
        tree_cnt = PointAddRangeSum(m)
        tree_sum = PointAddRangeSum(m)
        cb = Combinatorics(m, mod)
        for i, num in enumerate(nums):
            cnt = cb.inv[i + 1]
            smaller = tree_cnt.range_sum(1, num - 1) if num else 0
            tot += num + smaller * 2 * num + 2 * tree_sum.range_sum(1, m + 1 - num)
            tot %= mod
            tree_sum.point_add(m + 1 - num, num)
            tree_cnt.point_add(num, 1)
            ac.st(tot * cnt * cnt % mod)
        return

    @staticmethod
    def abc_256f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc256/tasks/abc256_f
        tag: tree_array|cumulative_cumulative_cumulative_sum|math|classical
        """
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        tree1 = PointAddRangeSum(n)
        tree1.build([nums[i - 1] * i * i for i in range(1, n + 1)])
        tree2 = PointAddRangeSum(n)
        tree2.build([nums[i - 1] * i for i in range(1, n + 1)])
        tree3 = PointAddRangeSum(n)
        tree3.build(nums)
        mod = 998244353
        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, v = lst[1:]
                p = v * x * x - nums[x - 1] * x * x
                tree1.point_add(x, p)
                p = v * x - nums[x - 1] * x
                tree2.point_add(x, p)
                p = v - nums[x - 1]
                tree3.point_add(x, p)
                nums[x - 1] = v
            else:
                x = lst[1]
                ans = (tree1.range_sum(1, x) - (2 * x + 3) * tree2.range_sum(1, x) + (x + 1) * (
                        x + 2) * tree3.range_sum(1, x)) // 2
                ans %= mod
                ac.st(ans)
        return

    @staticmethod
    def abc_250e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc250/tasks/abc250_e
        tag: tree_array|point_ascend|pre_max|implemention|set|classical
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        ind = dict()
        for i in range(n - 1, -1, -1):
            ind[a[i]] = i
        b = ac.read_list_ints()
        pre_a = [0] * n
        tmp = set()
        for i in range(n):
            tmp.add(a[i])
            pre_a[i] = len(tmp)

        pre_b = [0] * n
        tmp = set()
        for i in range(n):
            tmp.add(b[i])
            pre_b[i] = len(tmp)

        tree = PointAscendPreMax(n)
        for i in range(n):
            tree.point_ascend(i + 1, ind.get(b[i], inf) + 1)
        for _ in range(ac.read_int()):
            x, y = ac.read_list_ints()
            if pre_a[x - 1] == pre_b[y - 1] and tree.pre_max(y) <= x:
                ac.yes()
            else:
                ac.no()
        return

    @staticmethod
    def abc_231f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc231/tasks/abc231_f
        tag: discretize|tree_array|inclusion_exclusion|two_pointer
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        tot = sorted(set(a + b))
        ind = {num: i for i, num in enumerate(tot)}
        a = [ind[x] for x in a]
        b = [ind[x] for x in b]
        m = len(tot)
        tree = PointAddRangeSum(m)
        ind = list(range(n))
        ind.sort(key=lambda it: a[it])
        ans = tot = 0
        k = -1
        for i in ind:
            while k + 1 < n and a[ind[k + 1]] <= a[i]:
                tree.point_add(b[ind[k + 1]] + 1, 1)
                tot += 1
                k += 1
            tree.point_add(b[i] + 1, -1)
            tot -= 1
            aa, bb = a[i], b[i]
            if bb == 0:
                ans += tot
            else:
                ans += tot - tree.range_sum(1, bb)
            tree.point_add(b[i] + 1, 1)
            tot += 1
        ans += n
        ac.st(ans)
        return

    @staticmethod
    def abc_351f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc351/tasks/abc351_f
        tag: tree_array|discretize|classical
        """
        ac.read_int()
        nums = ac.read_list_ints()
        ind = {num: i + 1 for i, num in enumerate(sorted(set(nums)))}

        m = len(ind)
        tree_sum = PointAddRangeSum(m)
        tree_cnt = PointAddRangeSum(m)
        ans = 0
        for num in nums:
            tree_cnt.point_add(ind[num], 1)
            tree_sum.point_add(ind[num], num)
            ans += tree_cnt.range_sum(1, ind[num]) * num - tree_sum.range_sum(1, ind[num])
        ac.st(ans)
        return

    @staticmethod
    def abc_221e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc221/tasks/abc221_e
        tag: tree_array|contribution_method
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        mod = 998244353
        pp = [1] * (n + 1)
        for i in range(1, n + 1):
            pp[i] = (pp[i - 1] * 2) % mod
        rev = [1] * (n + 1)
        x = pow(2, -1, mod)
        for i in range(1, n + 1):
            rev[i] = (rev[i - 1] * x) % mod
        ind = {num: i + 1 for i, num in enumerate(sorted(set(nums)))}
        m = len(ind)
        tree = PointAddRangeSum(m)
        ans = 0
        for i in range(n):
            j = ind[nums[i]]
            if i:
                ans += tree.range_sum(1, j) * pp[i - 1] % mod
                ans %= mod
            tree.point_add(j, rev[i])
        ac.st(ans)
        return

    @staticmethod
    def abc_353g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc353/tasks/abc353_g
        tag: point_ascend|range_max|pre_max|classical
        """
        n, c = ac.read_list_ints()
        tree_ceil = PointAscendPreMax(n, -inf)
        tree_ceil.point_ascend(1, c)
        ceil = [-inf] * (n + 1)
        ceil[1] = c

        tree_floor = PointAscendPreMax(n, -inf)
        tree_floor.point_ascend(n, -c)

        ans = 0
        for _ in range(ac.read_int()):
            t, p = ac.read_list_ints()
            cur = ceil[t] - c * t + p
            if t > 1:
                cur = max(cur, tree_ceil.pre_max(t - 1) - c * t + p)
            if t + 1 < n:
                cur = max(cur, tree_floor.pre_max(n - t) + c * t + p)
            ans = max(ans, cur)
            ceil[t] = max(ceil[t], cur + c * t)
            tree_ceil.point_ascend(t, cur + c * t)
            tree_floor.point_ascend(n + 1 - t, cur - c * t)
        ac.st(ans)
        return

    @staticmethod
    def abc_356f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc356/tasks/abc356_f
        tag: tree_array|binary_search|bisect_right|classical
        """
        q, k = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(q)]
        nodes = sorted(set([x for _, x in nums] + [-1, 10 ** 18 + 1]))
        ind = {num: i for i, num in enumerate(nodes)}
        n = len(ind)
        tree_cnt = PointAddRangeSum(n)
        tree_root = PointAddRangeSum(n)  # index is point and value is root or not
        cur = set()
        root = set()
        part = 0
        for op, x in nums:
            i = ind[x]
            if op == 1:
                if x in cur:
                    tree_cnt.point_add(i + 1, -1)
                    cur.remove(x)
                    loc = tree_cnt.range_sum(1, i + 1)
                    left = -inf
                    right = inf
                    if loc:
                        left = nodes[tree_cnt.bisect_right(loc - 1)]
                    if loc < len(cur):
                        right = nodes[tree_cnt.bisect_right(loc)]
                    if x - left <= k < right - left:
                        tree_root.point_add(ind[left] + 1, 1)
                        root.add(ind[left] + 1)
                    if right - x > k:
                        tree_root.point_add(i + 1, -1)
                        root.remove(i + 1)
                else:
                    cur.add(x)
                    tree_cnt.point_add(i + 1, 1)
                    loc = tree_cnt.range_sum(1, i + 1)
                    left = -inf
                    right = inf
                    if loc > 1:
                        left = nodes[tree_cnt.bisect_right(loc - 2)]
                    if loc < len(cur):
                        right = nodes[tree_cnt.bisect_right(loc)]
                    if x - left <= k < right - left:
                        tree_root.point_add(ind[left] + 1, -1)
                        root.remove(ind[left] + 1)
                    if right - x > k:
                        tree_root.point_add(i + 1, 1)
                        root.add(i + 1)
                        part += 1
            else:
                loc = tree_root.range_sum(1, i + 1)
                label = 1 if i + 1 in root else 0
                if not label:
                    if loc:
                        left = tree_root.bisect_right(loc - 1) + 1
                    else:
                        left = 0
                    right = tree_root.bisect_right(loc)
                else:
                    if loc >= 2:
                        left = tree_root.bisect_right(loc - 2) + 1
                    else:
                        left = 0
                    right = i
                ans = tree_cnt.range_sum(left + 1, right + 1)
                ac.st(ans)
        return

    @staticmethod
    def lc_100317(nums: List[int], queries: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/peaks-in-array/
        tag: tree_array|implemention
        """
        n = len(nums)
        tree = PointAddRangeSum(n)
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                tree.point_add(i, 1)

        res = []
        for op, a, b in queries:
            if op == 1:
                res.append(tree.range_sum(a + 1, b - 1) if a + 1 <= b - 1 else 0)
            else:
                for aa in [a - 1, a, a + 1]:
                    if 0 <= aa <= aa + 1 < n and nums[aa] > nums[aa - 1] and nums[aa] > nums[aa + 1]:
                        tree.point_add(aa, -1)
                nums[a] = b
                for aa in [a - 1, a, a + 1]:
                    if 0 <= aa < aa + 1 < n and nums[aa] > nums[aa - 1] and nums[aa] > nums[aa + 1]:
                        tree.point_add(aa, 1)
        return res

    @staticmethod
    def cf_1430e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1430/E
        tag: tree_array|classical|implemention|point_add|range_sum|pre_sum
        """
        n = ac.read_int()
        s = ac.read_str()
        dct = defaultdict(list)
        for i in range(n - 1, -1, -1):
            dct[s[i]].append(i)
        tree = PointAddRangeSum(n)
        tree.build([1] * n)
        ans = 0
        for i in range(n - 1, -1, -1):
            x = dct[s[i]].pop()
            ans += tree.range_sum(0, x) - 1
            tree.point_add(x, -1)
        ac.st(ans)
        return

    @staticmethod
    def cf_1788e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1788/E
        tag: linear_dp|tree_array|point_ascend|pre_max
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = ac.accumulate(nums)
        nodes = sorted(set(pre))
        ind = {num: i + 1 for i, num in enumerate(nodes)}
        m = len(ind)
        tree = PointAscendPreMax(m)
        pre_max = [0] * (n + 1)
        tree.point_ascend(ind[0], 0)
        for i in range(n):
            cur = tree.pre_max(ind[pre[i + 1]]) + i + 1
            pre_max[i + 1] = max(pre_max[i], cur)
            tree.point_ascend(ind[pre[i + 1]], pre_max[i + 1] - i - 1)
        ans = pre_max[-1]
        ac.st(ans)
        return

    @staticmethod
    def cf_677d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/677/D
        tag: layered_bfs|tree_array|two_pointer|partial_order|implemention|classical
        """
        m, n, p = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        dct = [[] for _ in range(p + 1)]
        for i in range(m):
            for j in range(n):
                dct[grid[i][j]].append(i * n + j)
        dct[1].sort()
        dis = [inf] * m * n
        for x in dct[1]:
            i, j = x // n, x % n
            dis[i * n + j] = i + j
        tree_pre = PointDescendPreMin(n)
        tree_post = PointDescendPostMin(n)
        for x in range(2, p + 1):
            dct[x].sort()
            k1 = len(dct[x - 1])
            k2 = len(dct[x])
            if k1 * k2 <= 100:
                for x1 in dct[x - 1]:
                    i1, j1 = x1 // n, x1 % n
                    for x2 in dct[x]:
                        i2, j2 = x2 // n, x2 % n
                        dis[x2] = min(dis[x2], dis[x1] + abs(i2 - i1) + abs(j2 - j1))
            else:
                ind = 0
                tree_pre.initialize()
                tree_post.initialize()
                for x2 in dct[x]:
                    i2, j2 = x2 // n, x2 % n
                    while ind < k1 and dct[x - 1][ind] // n <= i2:
                        x1 = dct[x - 1][ind]
                        i1, j1 = x1 // n, x1 % n
                        tree_pre.point_descend(j1, -i1 - j1 + dis[x1])
                        tree_post.point_descend(j1, j1 - i1 + dis[x1])
                        ind += 1
                    cur = min(i2 + j2 + tree_pre.pre_min(j2), i2 - j2 + tree_post.post_min(j2))
                    dis[x2] = cur

                ind = k1 - 1
                tree_pre.initialize()
                tree_post.initialize()
                for x2 in dct[x][::-1]:
                    i2, j2 = x2 // n, x2 % n
                    while ind >= 0 and dct[x - 1][ind] // n >= i2:
                        x1 = dct[x - 1][ind]
                        i1, j1 = x1 // n, x1 % n
                        tree_pre.point_descend(j1, i1 - j1 + dis[x1])
                        tree_post.point_descend(j1, j1 + i1 + dis[x1])
                        ind -= 1
                    cur = min(j2 - i2 + tree_pre.pre_min(j2), - i2 - j2 + tree_post.post_min(j2))
                    dis[x2] = min(dis[x2], cur)
        ans = min(dis[x] for x in dct[p])
        ac.st(ans)
        return

    @staticmethod
    def lg_p5041(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5041
        tag: tree_array|implemention|classical
        """
        s = ac.read_str()
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
                ans += left - i - tree.range_sum(i, left)
                x = dct[lst[j]].popleft()
                dct[lst[j]].pop()
                lst[x] = ""
                tree.point_add(x, 1)
                j -= 1
            else:
                right = dct[lst[i]][-1]
                ans += j - right - tree.range_sum(right, j)
                x = dct[lst[i]].pop()
                dct[lst[i]].popleft()
                tree.point_add(x, 1)
                lst[x] = ""
                i += 1
        ac.st(ans)
        return

    @staticmethod
    def abc_368g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc368/tasks/abc368_g
        tag: point_add|range_sum|observation|data_range
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        tree = PointAddRangeSum(n)
        tree.build(a)
        lst = SortedList([i for i in range(n) if b[i] > 1] + [n])

        for _ in range(ac.read_int()):
            op, i, x = ac.read_list_ints()
            if op == 1:
                tree.point_add(i - 1, x - a[i - 1])
                a[i - 1] = x
            elif op == 2:
                if b[i - 1] != 1:
                    lst.discard(i - 1)
                b[i - 1] = x
                if b[i - 1] != 1:
                    lst.add(i - 1)
            else:
                v = 0
                i -= 1
                x -= 1
                j = lst.bisect_left(i)
                while i <= x:
                    while lst[j] < i:
                        j += 1
                    if lst[j] == i:
                        v = max(v + a[lst[j]], v * b[lst[j]])
                        i += 1
                    elif lst[j] <= x:
                        v += tree.range_sum(i, lst[j] - 1)
                        v = max(v + a[lst[j]], v * b[lst[j]])
                        i = lst[j] + 1
                    else:
                        v += tree.range_sum(i, x)
                        i = x + 1
                ac.st(v)
        return

    @staticmethod
    def abc_369f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc369/tasks/abc369_f
        tag: tree_array|point_ascend|pre_max_index|construction|specific_plan
        """
        m, n, k = ac.read_list_ints()
        nums = [ac.read_list_ints_minus_one() for _ in range(k)]

        tree = PointAscendPreMaxIndex(2 * 10 ** 5, 0)
        ind = list(range(k))
        ind.sort(key=lambda it: nums[it])
        dct = dict()
        for i in ind:
            x, y = nums[i]
            pre, ind = tree.pre_max(y)
            dct[i] = ind
            tree.point_ascend(y, pre + 1, i)

        ans1, ans2 = tree.pre_max(2 * 10 ** 5 - 1)
        path = [ans2]
        while path[-1] != -1:
            path.append(dct[path[-1]])
        path.pop()
        path.reverse()
        path = [(0, 0)] + [nums[x] for x in path] + [(m - 1, n - 1)]
        lst = []
        for i in range(1, len(path)):
            a, b = path[i - 1]
            c, d = path[i]
            lst.append("R" * (d - b))
            lst.append("D" * (c - a))
        ac.st(ans1)
        ac.st("".join(lst))
        return

    @staticmethod
    def cf_1667b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1667/B
        tag: tree_array|classical|prefix_sum
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            pre = ac.accumulate(nums)
            ind = {num: i for i, num in enumerate(sorted(set(pre)))}
            m = len(ind)
            tree_ceil = PointAscendPreMax(m)
            tree_floor = PointAscendPreMax(m)
            ans = 0
            tree_ceil.point_ascend(ind[0], 0)
            tree_floor.point_ascend(m - ind[0] - 1, 0)
            ac.get_random_seed()
            dct = dict()
            dct[0 ^ ac.random_seed] = 0
            for i in range(n):
                ans = dct.get(pre[i + 1] ^ ac.random_seed, -inf)
                # < pre[i+1]
                cur = tree_ceil.pre_max(ind[pre[i + 1]] - 1) if ind[pre[i + 1]] >= 1 else -inf
                ans = max(ans, cur + i + 1)
                # > pre[i+1]
                cur = tree_floor.pre_max(m - 1 - ind[pre[i + 1]] - 1) if m - 1 - ind[pre[i + 1]] >= 1 else -inf
                ans = max(ans, cur - i - 1)
                # update
                tree_floor.point_ascend(m - 1 - ind[pre[i + 1]], ans + (i + 1))
                tree_ceil.point_ascend(ind[pre[i + 1]], ans - (i + 1))
                dct[pre[i + 1] ^ ac.random_seed] = max(ans, dct.get(pre[i + 1] ^ ac.random_seed, -inf))
            ac.st(ans)
        return
