"""
Algorithm：matrix_dp|memory_search|lcs|md_matrix_dp
Description：matrix_prefix_sum|sub_matrix_sum|maximum_square|edit_distance|lcs|palindrome_substring

====================================LeetCode====================================
174（https://leetcode.cn/problems/dungeon-game/）matrix_dp|reverse_thinking
2478（https://leetcode.cn/problems/number-of-beautiful-partitions/）matrix_dp
2463（https://leetcode.cn/problems/minimum-total-distance-traveled/）matrix_dp
2435（https://leetcode.cn/problems/paths-in-matrix-whose-sum-is-divisible-by-k/）matrix_dp|mod
2088（https://leetcode.cn/problems/count-fertile-pyramids-in-a-land/）matrix_dp
221（https://leetcode.cn/problems/maximal-square/）maximum_square|classical
72（https://leetcode.cn/problems/edit-distance/）matrix_dp
329（https://leetcode.cn/problems/longest-increasing-path-in-a-matrix/）matrix_dp
1478（https://leetcode.cn/problems/allocate-mailboxes/）matrix_dp|median_greedy|interval_dp|greedy
2573（https://leetcode.cn/problems/find-the-string-with-lcp/）greedy|construction|lcp
2328（https://leetcode.cn/problems/number-of-increasing-paths-in-a-grid/）matrix_dp|counter
2312（https://leetcode.cn/problems/selling-pieces-of-wood/）memory_search|specific_plan
2267（https://leetcode.cn/problems/check-if-there-is-a-valid-parentheses-string-path/）memory_search
1092（https://leetcode.cn/problems/shortest-common-supersequence/）construction|lcs|construction|specific_plan
1143（https://leetcode.cn/problems/longest-common-subsequence/）lis|lcs
1035（https://leetcode.cn/problems/uncrossed-lines/）lis|lcs
2617（https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/）reverse_order|matrix_dp|tree_array|prefix_min
1092（https://leetcode.cn/problems/shortest-common-supersequence/）lis|lcs|specific_plan
1692（https://leetcode.cn/problems/count-ways-to-distribute-candies/）matrix_dp|specific_plan|counter
1771（https://leetcode.cn/problems/maximize-palindrome-length-from-subsequences/）longest_palindrome_subsequence|matrix_dp
1883（https://leetcode.cn/problems/minimum-skips-to-arrive-at-meeting-on-time/）matrix_dp|high_precision|bag_dp
1977（https://leetcode.cn/problems/number-of-ways-to-separate-numbers/）matrix_dp|lcp|prefix_sum
2430（https://leetcode.cn/problems/maximum-deletions-on-a-string/）lcp|matrix_dp
1216（https://leetcode.cn/problems/valid-palindrome-iii/）matrix_dp|longest_palindrome_subsequence
2060（https://leetcode.cn/problems/check-if-an-original-string-exists-given-two-encoded-strings/description/）matrix_dp|brute_force|memory_search
2556（https://leetcode.cn/problems/disconnect-path-in-a-binary-matrix-by-at-most-one-flip/description/）matrix_dp|brain_teaser
920（https://leetcode.cn/problems/number-of-music-playlists/）matrix_dp
1594（https://leetcode.cn/problems/maximum-non-negative-product-in-a-matrix/）matrix_dp|maximum_mul|minimum_mul
1639（https://leetcode.cn/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/）prefix_sum|matrix_dp
956（https://leetcode.cn/problems/tallest-billboard/description/）matrix_dp
1301（https://leetcode.cn/contest/biweekly-contest-16/problems/number-of-paths-with-max-score/）matrix_dp|specific_plan|counter
1937（https://leetcode.cn/problems/maximum-number-of-points-with-cost/）prefix_sum|matrix_dp
1751（https://leetcode.cn/problems/maximum-number-of-events-that-can-be-attended-ii/）matrix_dp|binary_search
1959（https://leetcode.cn/problems/minimum-total-space-wasted-with-k-resizing-operations/description/）matrix_dp|prefix_sum
1458（https://leetcode.cn/problems/max-dot-product-of-two-subsequences/description/）matrix_dp
1745（https://leetcode.cn/problems/palindrome-partitioning-iv/description/）matrix_dp|palindrome_substring|manacher|brute_force
2809（https://leetcode.cn/problems/minimum-time-to-make-array-sum-at-most-x/）matrix_dp|greedy|implemention

=====================================LuoGu======================================
P2701（https://www.luogu.com.cn/problem/P2701）maximum_square|matrix_dp|brute_force|classical|O(n^3)|hollow
P2049（https://www.luogu.com.cn/problem/P2049）matrix_dp|counter|mod
P2138（https://www.luogu.com.cn/problem/P2138）matrix_dp|lcs
P1681（https://www.luogu.com.cn/problem/P1681）maximum_square|matrix_dp
P2268（https://www.luogu.com.cn/problem/P2268）matrix_dp|edit_distance
P2301（https://www.luogu.com.cn/problem/P2301）matrix_dp
P2364（https://www.luogu.com.cn/problem/P2364）lcs|specific_plan|matrix_dp
P2543（https://www.luogu.com.cn/problem/P2543）matrix_dp|lcs
list?user=739032&status=12&page=2（https://www.luogu.com.cn/record/list?user=739032&status=12&page=2）matrix_dp|prefix_sum
P1434（https://www.luogu.com.cn/problem/P1434）matrix_dp|lis
P1140（https://www.luogu.com.cn/problem/P1140）matrix_dp
P1057（https://www.luogu.com.cn/problem/P1057）matrix_dp
P8825（https://www.luogu.com.cn/problem/P8825）mod|matrix_dp|rolling_update
P2758（https://www.luogu.com.cn/problem/P2758）matrix_dp|edit_distance
P2803（https://www.luogu.com.cn/problem/P2803）matrix_dp|median_greedy|interval_dp
P2946（https://www.luogu.com.cn/problem/P2946）matrix_dp
P2427（https://www.luogu.com.cn/problem/P2427）matrix_dp|square
P7074（https://www.luogu.com.cn/problem/P7074）matrix_dp
P7160（https://www.luogu.com.cn/problem/P7160）matrix_dp|brute_force|counter
P7266（https://www.luogu.com.cn/problem/P7266）matrix_dp
P3399（https://www.luogu.com.cn/problem/P3399）matrix_dp
P2516（https://www.luogu.com.cn/problem/P2516）lcs|matrix_dp
P1544（https://www.luogu.com.cn/problem/P1544）matrix_dp
P1004（https://www.luogu.com.cn/problem/P1004）matrix_dp
P1006（https://www.luogu.com.cn/problem/P1006）matrix_dp
P1107（https://www.luogu.com.cn/problem/P1107）matrix_dp|prefix_max
P1279（https://www.luogu.com.cn/problem/P1279）edit_distance
P1353（https://www.luogu.com.cn/problem/P1353）matrix_dp
P1410（https://www.luogu.com.cn/problem/P1410）matrix_dp
P1799（https://www.luogu.com.cn/problem/P1799）matrix_dp
P1854（https://www.luogu.com.cn/problem/P1854）prefix_max|matrix_dp|specific_plan
P2140（https://www.luogu.com.cn/problem/P2140）matrix_dp
P2217（https://www.luogu.com.cn/problem/P2217）matrix_dp
P1436（https://www.luogu.com.cn/problem/P1436）md_matrix_dp
P5752（https://www.luogu.com.cn/problem/P5752）md_matrix_dp
P2380（https://www.luogu.com.cn/problem/P2380）md_matrix_dp
P2401（https://www.luogu.com.cn/problem/P2401）md_matrix_dp
P2528（https://www.luogu.com.cn/problem/P2528）reverse_order_pair|matrix_dp|implemention|construction
P2733（https://www.luogu.com.cn/problem/P2733）diff_array|matrix_dp|counter|maximum_square
P2736（https://www.luogu.com.cn/problem/P2736）matrix_dp
P2769（https://www.luogu.com.cn/problem/P2769）matrix_dp
P3012（https://www.luogu.com.cn/problem/P3012）matrix_dp
P3860（https://www.luogu.com.cn/problem/P3860）matrix_dp|specific_plan
P4958（https://www.luogu.com.cn/problem/P4958）linear_dp|prefix_sum
P5144（https://www.luogu.com.cn/problem/P5144）linear_dp|prefix_xor
P5858（https://www.luogu.com.cn/problem/P5858）matrix_dp|monotonic_queue
P5879（https://www.luogu.com.cn/problem/P5879）matrix_dp|prefix_sum
P6119（https://www.luogu.com.cn/problem/P6119）matrix_dp|lcs
P6323（https://www.luogu.com.cn/problem/P6323）reverse_order_pair|prefix_sum
P6394（https://www.luogu.com.cn/problem/P6394）matrix_dp|prefix_sum
P6433（https://www.luogu.com.cn/problem/P6433）greedy|classification_discussion|matrix_dp
P6451（https://www.luogu.com.cn/problem/P6451）md_matrix_dp|brute_force|4-tree
P6509（https://www.luogu.com.cn/problem/P6509）classical|matrix_dp|specific_plan
P6870（https://www.luogu.com.cn/problem/P6870）matrix_dp|comb|counter
P7995（https://www.luogu.com.cn/problem/P7995）matrix_dp
P8325（https://www.luogu.com.cn/problem/P8325）brute_force|matrix_dp|maximum_square
P8614（https://www.luogu.com.cn/problem/P8614）matrix_dp|mod
P8638（https://www.luogu.com.cn/problem/P8638）matrix_dp|longest_palindrome_sequence
P8786（https://www.luogu.com.cn/problem/P8786）classical|md_matrix_dp|implemention|memory_search

===================================CodeForces===================================
1446B（https://codeforces.com/problemset/problem/1446/B）lcs|matrix_dp
429B（https://codeforces.com/problemset/problem/429/B）matrix_dp
1398D（https://codeforces.com/problemset/problem/1398/D）md_matrix_dp|maximum_mul|maximum_sum
2B（https://codeforces.com/problemset/problem/2/B）matrix_dp
1381B（https://codeforces.com/problemset/problem/1381/B）matrix_dp|monotonic_stack
1393D（https://codeforces.com/problemset/problem/1393/D）matrix_dp
1731D（https://codeforces.com/contest/1731/problem/D）binary_search|maximum_square
1003F（https://codeforces.com/contest/1003/problem/F）con_lcp|matrix_dp|lcp
835D（https://codeforces.com/problemset/problem/835/D）palindrome|matrix_dp
1829G（https://codeforces.com/contest/1829/problem/G）matrix_dp|classical|inclusion_exclusion
1077F2（https://codeforces.com/contest/1077/problem/F2）matrix_dp|monotonic_queue|implemention
1133E（https://codeforces.com/contest/1133/problem/E）matrix_dp|preprocess|classical
1183H（https://codeforces.com/contest/1183/problem/H）matrix_dp|classical|hard|different_sub_sequence
1183E（https://codeforces.com/contest/1183/problem/E）matrix_dp|classical|hard|different_sub_sequence
1353F（https://codeforces.com/contest/1353/problem/F）matrix_dp|greedy|monotonic_stack
1409F（https://codeforces.com/contest/1409/problem/F）matrix_dp
1433F（https://codeforces.com/contest/1433/problem/F）matrix_dp
1551E（https://codeforces.com/contest/1551/problem/E）matrix_dp
1593F（https://codeforces.com/contest/1593/problem/F）matrix_dp|specific_plan|md_vector|flatten
1729G（https://codeforces.com/contest/1729/problem/G）matrix_dp
1811G2（https://codeforces.com/contest/1811/problem/G2）matrix_dp|comb

====================================AtCoder=====================================
ABC130E（https://atcoder.jp/contests/abc130/tasks/abc130_e）matrix_prefix_sum|matrix_dp
ABC325F（https://atcoder.jp/contests/abc325/tasks/abc325_f）matrix_dp|brute_force|classical
ABC344F（https://atcoder.jp/contests/abc344/tasks/abc344_f）matrix_dp|greedy|brain_teaser|classical
ABC311F（https://atcoder.jp/contests/abc311/tasks/abc311_f）matrix_dp|prefix_sum_opt|classical|brain_teaser
ABC311E（https://atcoder.jp/contests/abc311/tasks/abc311_e）matrix_dp|classical
ABC298G（https://atcoder.jp/contests/abc298/tasks/abc298_g）matrix_dp|brute_force|classical
ABC281D（https://atcoder.jp/contests/abc281/tasks/abc281_d）matrix_dp
ABC265E（https://atcoder.jp/contests/abc265/tasks/abc265_e）matrix_dp|brain_teaser|classical
ABC264F（https://atcoder.jp/contests/abc264/tasks/abc264_f）matrix_dp|tle
ABC261D（https://atcoder.jp/contests/abc261/tasks/abc261_d）matrix_dp
ABC262D（https://atcoder.jp/contests/abc262/tasks/abc262_d）brute_force|matrix_dp|classical
ABC253E（https://atcoder.jp/contests/abc253/tasks/abc253_e）prefix_sum|matrix_dp|inclusion_exclusion|reverse_thinking|classical

=====================================AcWing=====================================
4378（https://www.acwing.com/problem/content/4381/）classical|matrix_dp
4418（https://www.acwing.com/problem/content/description/4421/）monotonic_queue|matrix_dp
2694（https://www.acwing.com/problem/content/description/2696/）lcs|matrix_dp|counter


"""

import heapq
import math
from collections import defaultdict, deque
from functools import lru_cache
from itertools import permutations, accumulate
from typing import List

from src.basis.diff_array.template import PreFixSumMatrix
from src.data_structure.tree_array.template import PointDescendPreMin
from src.greedy.longest_increasing_subsequence.template import LcsComputeByLis
from src.mathmatics.comb_perm.template import Combinatorics
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1305(nums1: List[int], nums2: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/uncrossed-lines/
        tag: lis|lcs
        """
        return LcsComputeByLis().length_of_lcs(nums1, nums2)

    @staticmethod
    def lc_1143(s1: str, s2: str) -> int:
        """
        url: https://leetcode.cn/problems/longest-common-subsequence/
        tag: lis|lcs
        """
        return LcsComputeByLis().length_of_lcs(s1, s2)

    @staticmethod
    def lc_920(n: int, goal: int, k: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-music-playlists/
        tag: matrix_dp|classical|hard|fill_table
        """
        mod = 10 ** 9 + 7

        @lru_cache(None)
        def dfs(i, r):
            if i == goal:
                return 1 if r == n else 0
            res = 0
            if r + 1 <= n:
                res += dfs(i + 1, r + 1) * (n - r)
            if r > k:
                res += dfs(i + 1, r) * (r - k)
            return res % mod

        return dfs(0, 0)

    @staticmethod
    def lc_956(rods: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/tallest-billboard/description/
        tag: matrix_dp|classical|meet_in_middle
        """
        pre = defaultdict(int)
        pre[0] = 0
        for num in rods:
            cur = pre.copy()
            for p in pre:
                cur[p + num] = max(cur[p + num], pre[p])
                cur[p - num] = max(cur[p - num], pre[p] + num)
            pre = cur
        return pre[0]

    @staticmethod
    def lc_1092(str1: str, str2: str) -> str:
        """
        url: https://leetcode.cn/problems/shortest-common-supersequence/
        tag: lis|lcs|specific_plan
        """
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                a, b = dp[i + 1][j], dp[i][j + 1]
                a = a if a > b else b
                if str1[i] == str2[j]:
                    b = dp[i + 1][j + 1] + 1
                    a = a if a > b else b
                dp[i][j] = a

        i = j = 0
        ans = ""
        while i < m and j < n:
            if str1[i] == str2[j]:
                ans += str1[i]
                i += 1
                j += 1
            elif dp[i + 1][j + 1] == dp[i + 1][j]:
                ans += str2[j]
                j += 1
            else:
                ans += str1[i]
                i += 1
        ans += str1[i:] + str2[j:]
        return ans

    @staticmethod
    def lc_2435(grid: List[List[int]], k: int) -> int:
        """
        url: https://leetcode.cn/problems/paths-in-matrix-whose-sum-is-divisible-by-k/
        tag: matrix_dp|mod
        """
        mod = 10 ** 9 + 7
        m, n = len(grid), len(grid[0])
        dp = [[[0] * k for _ in range(n)] for _ in range(m)]
        dp[0][0][grid[0][0] % k] = 1

        pre = grid[0][0]
        for j in range(1, n):
            pre += grid[0][j]
            dp[0][j][pre % k] = 1

        pre = grid[0][0]
        for i in range(1, m):
            pre += grid[i][0]
            dp[i][0][pre % k] = 1

        for i in range(1, m):
            for j in range(1, n):
                for x in range(k):
                    y = (x - grid[i][j]) % k
                    dp[i][j][x] = (dp[i - 1][j][y] + dp[i][j - 1][y]) % mod
        return dp[-1][-1][0]

    @staticmethod
    def lc_2573(lcp: List[List[int]]) -> str:
        """
        url: https://leetcode.cn/problems/find-the-string-with-lcp/
        tag: greedy|construction|lcp|brain_teaser|classical
        """
        n = len(lcp)
        ans = [""] * n
        ind = 0
        for i in range(n):
            if ans[i]:
                continue
            if ind == 26:
                return ""
            w = chr(ind + ord("a"))
            ans[i] = w
            ind += 1
            for j in range(i + 1, n):
                if lcp[i][j]:
                    ans[j] = w

        ans = "".join(ans)
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if ans[i] == ans[j]:
                    if i + 1 < n and j + 1 < n:
                        x = lcp[i + 1][j + 1] + 1
                    else:
                        x = 1
                else:
                    x = 0
                if x != lcp[i][j]:
                    return ""
        return ans

    @staticmethod
    def cf_2b(ac, n, grid):
        """
        url: https://codeforces.com/problemset/problem/2/B
        tag: matrix_dp
        """

        def f_2(num):
            if not num:
                return 1
            res = 0
            while num and num % 2 == 0:
                num //= 2
                res += 1
            return res

        def f_5(num):
            if not num:
                return 1
            res = 0
            while num and num % 5 == 0:
                num //= 5
                res += 1
            return res

        def check(fun):
            dp = [[inf] * n for _ in range(n)]
            dp[0][0] = fun(grid[0][0])
            f = [[-1] * n for _ in range(n)]
            for j in range(1, n):
                f[0][j] = j - 1
                dp[0][j] = dp[0][j - 1] + fun(grid[0][j]) if grid[0][j] else 1
            for i in range(1, n):
                f[i][0] = (i - 1) * n
                dp[i][0] = dp[i - 1][0] + fun(grid[i][0]) if grid[i][0] else 1
                for j in range(1, n):
                    if grid[i][j] == 0:
                        dp[i][j] = 1
                    else:
                        c = fun(grid[i][j])
                        dp[i][j] = ac.min(dp[i - 1][j], dp[i][j - 1]) + c
                    f[i][j] = (i - 1) * n + j if dp[i - 1][j] < dp[i][j - 1] else i * n + j - 1
            cnt = dp[-1][-1]
            path = ""
            x = (n - 1) * n + n - 1
            while f[x // n][x % n] != -1:
                i, j = x // n, x % n
                p = f[i][j]
                if i == p // n:
                    path += "R"
                else:
                    path += "D"
                x = p
            return cnt, path[::-1]

        c1, path1 = check(f_2)
        c2, path2 = check(f_5)
        if c1 <= c2:
            ans = [c1, path1]
        else:
            ans = [c2, path2]

        zero = False
        for ii in range(n):
            for jj in range(n):
                if grid[ii][jj] == 0:
                    zero = True
        if not zero:
            return ans

        if ans[0] > 1:
            for ii in range(n):
                for jj in range(n):
                    if grid[ii][jj] == 0:
                        cur = "D" * ii + "R" * jj + "D" * (n - 1 - ii) + "R" * (n - 1 - jj)
                        ans = [1, cur]
                        return ans
        return ans

    @staticmethod
    def cf_1398d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1398/D
        tag: md_matrix_dp|maximum_mul|maximum_sum
        """
        r, g, b = ac.read_list_ints()
        rr = sorted(ac.read_list_ints(), reverse=True)
        gg = sorted(ac.read_list_ints(), reverse=True)
        bb = sorted(ac.read_list_ints(), reverse=True)

        def idx(i1, i2, i3):
            return i1 * (g + 1) * (b + 1) + i2 * (b + 1) + i3

        dp = [0] * (r + 1) * (g + 1) * (b + 1)

        for i in range(r, -1, -1):
            for j in range(g, -1, -1):
                for k in range(b, -1, -1):
                    res = 0
                    if i < r and j < g:
                        res = ac.max(res, dp[idx(i + 1, j + 1, k)] + rr[i] * gg[j])
                    if i < r and k < b:
                        res = ac.max(res, dp[idx(i + 1, j, k + 1)] + rr[i] * bb[k])
                    if j < g and k < b:
                        res = ac.max(res, dp[idx(i, j + 1, k + 1)] + bb[k] * gg[j])
                    dp[idx(i, j, k)] = res

        ac.st(dp[0])
        return

    @staticmethod
    def lc_2478(s: str, k: int, min_length: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-beautiful-partitions/
        tag: matrix_dp|classical|rever_thinking
        """
        mod = 10 ** 9 + 7
        n = len(s)
        prime = set("2357")
        if not (s[0] in prime and s[-1] not in prime):
            return 0
        if k == 1:
            return 1

        cut = []
        for i in range(min_length - 1, n - min_length):
            if s[i] not in prime and s[i + 1] in prime:
                cut.append(i)

        m = len(cut)
        if m + 1 < k:
            return 0

        pre = [1] * m

        for _ in range(2, k):
            cur = [0] * m
            x = j = 0
            for i in range(m):
                while j < m and cut[i] - cut[j] >= min_length:
                    x += pre[j]
                    x %= mod
                    j += 1
                cur[i] = x
            pre = cur[:]
        return sum(pre) % mod

    @staticmethod
    def lc_2463(robot, factory):
        """
        url: https://leetcode.cn/problems/minimum-total-distance-traveled/
        tag: matrix_dp|refresh_table
        """
        robot.sort()
        factory.sort()
        m, n = len(factory), len(robot)
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(m):
            for j in range(n + 1):
                if dp[i][j] < inf:
                    dp[i + 1][j] = min(dp[i + 1][j], dp[i][j])
                    cost = 0
                    for k in range(1, factory[i][1] + 1):
                        if j + k - 1 < n:
                            cost += abs(factory[i][0] - robot[j + k - 1])
                            dp[i + 1][j + k] = min(dp[i + 1][j + k], dp[i][j] + cost)
                        else:
                            break
        return dp[-1][-1]

    @staticmethod
    def lg_p2516(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2516
        tag: lcs|matrix_dp|length_of_lcs|cnt_of_lcs|rolling_array|classical|hard
        """
        s = ac.read_str()[:-1]
        t = ac.read_str()[:-1]
        m, n = len(s), len(t)
        mod = 10 ** 8
        pre_dp = [0] * (n + 1)
        pre_cnt = [1] * (n + 1)
        cur_dp = [0] * (n + 1)
        cur_cnt = [0] * (n + 1)
        pre = 0
        for i in range(m):
            cur = 1 - pre
            cur_dp[0] = 0
            cur_cnt[0] = 1
            for j in range(n):
                cur_dp[j + 1] = 0
                cur_cnt[j + 1] = 0

                if s[i] == t[j]:
                    cur_dp[j + 1] = pre_dp[j] + 1
                    cur_cnt[j + 1] = pre_cnt[j]

                if cur_dp[j] > cur_dp[j + 1]:
                    cur_dp[j + 1] = cur_dp[j]
                    cur_cnt[j + 1] = cur_cnt[j]
                elif cur_dp[j] == cur_dp[j + 1]:
                    cur_cnt[j + 1] += cur_cnt[j]

                if pre_dp[j + 1] > cur_dp[j + 1]:
                    cur_dp[j + 1] = pre_dp[j + 1]
                    cur_cnt[j + 1] = pre_cnt[j + 1]
                elif pre_dp[j + 1] == cur_dp[j + 1]:
                    cur_cnt[j + 1] += pre_cnt[j + 1]

                if pre_dp[j] == cur_dp[j + 1]:
                    cur_cnt[j + 1] -= pre_cnt[j]
                cur_cnt[j + 1] %= mod
            for j in range(n + 1):
                pre_dp[j] = cur_dp[j]
                pre_cnt[j] = cur_cnt[j]
            pre = cur

        ac.st(pre_dp[-1])
        ac.st(pre_cnt[-1])
        return

    @staticmethod
    def lg_p1544(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1544
        tag: matrix_dp
        """
        n, k = ac.read_list_ints()

        pre = [-inf] * (k + 1) * n
        cur = [-inf] * (k + 1) * n
        pre[0] = 0
        for i in range(1, n + 1):
            lst = ac.read_list_ints()
            for j in range(i):
                for p in range(k + 1):
                    if j and p:
                        a = ac.max(pre[j * (k + 1) + p], pre[(j - 1) * (k + 1) + p]) + lst[j]
                        b = ac.max(pre[j * (k + 1) + p - 1], pre[(j - 1) * (k + 1) + p - 1]) + lst[j] * 3
                        cur[j * (k + 1) + p] = ac.max(a, b)
                    elif j:
                        cur[j * (k + 1) + p] = ac.max(pre[j * (k + 1) + p], pre[(j - 1) * (k + 1) + p]) + lst[j]
                    elif p:
                        cur[j * (k + 1) + p] = ac.max(pre[j * (k + 1) + p] + lst[j],
                                                      pre[j * (k + 1) + p - 1] + lst[j] * 3)
                    else:
                        cur[j * (k + 1) + p] = pre[j * (k + 1) + p] + lst[j]
            for j in range(n * (k + 1)):
                pre[j] = cur[j]
        ac.st(max(pre))
        return

    @staticmethod
    def lg_p1004(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1004
        tag: matrix_dp|classical
        """
        n = ac.read_int()
        grid = [[0] * n for _ in range(n)]
        while True:
            lst = ac.read_list_ints()
            if lst == [0, 0, 0]:
                break
            x, y, z = lst
            grid[x - 1][y - 1] = z
        pos = [[] for _ in range(2 * n - 1)]
        for i in range(n):
            for j in range(n):
                pos[i + j].append(i)

        pre = {(0, 0): grid[0][0]}
        for i in range(1, 2 * n - 1):
            cur = dict()
            for x1 in pos[i]:
                for x2 in pos[i]:
                    val = 0
                    y1, y2 = i - x1, i - x2
                    for a, b in [[x1 - 1, y1], [x1, y1 - 1]]:
                        for c, d in [[x2 - 1, y2], [x2, y2 - 1]]:
                            if 0 <= a < n and 0 <= b < n and 0 <= c < n and 0 <= d < n:
                                val = ac.max(val, pre[(b, d)])
                    val += grid[x1][y1] + grid[x2][y2]
                    if x1 == x2:
                        val -= grid[x1][y1]
                    cur[(y1, y2)] = val
            pre = cur
        ac.st(list(pre.values())[0])
        return

    @staticmethod
    def lg_p1006(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1006
        tag: matrix_dp
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]

        pos = [[] for _ in range(m + n - 1)]
        for i in range(m):
            for j in range(n):
                pos[i + j].append(i)

        pre = {(0, 0): grid[0][0]}
        for i in range(1, m + n - 1):
            cur = dict()
            for x1 in pos[i]:
                for x2 in pos[i]:
                    val = 0
                    y1, y2 = i - x1, i - x2
                    for a, b in [[x1 - 1, y1], [x1, y1 - 1]]:
                        for c, d in [[x2 - 1, y2], [x2, y2 - 1]]:
                            if 0 <= a < m and 0 <= b < n and 0 <= c < m and 0 <= d < n:
                                val = ac.max(val, pre[(b, d)])
                    val += grid[x1][y1] + grid[x2][y2]
                    if y1 == y2 and i < m + n - 2:
                        val = -inf
                    cur[(y1, y2)] = val
            pre = cur
        ac.st(list(pre.values())[0])
        return

    @staticmethod
    def lg_p1107(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1107
        tag: matrix_dp|prefix_max
        """
        n, h, d = ac.read_list_ints()
        cnt = [0] * (h + 1) * n
        for i in range(n):
            lst = ac.read_list_ints()
            for j in lst[1:]:
                cnt[i * (h + 1) + j] += 1

        ceil = [0] * (h + 1)
        pre = [0] * n
        cur = [0] * n
        for i in range(h, -1, -1):
            for j in range(n):
                cur[j] = pre[j] + cnt[j * (h + 1) + i]
                if i + d <= h and ceil[i + d] > pre[j]:
                    cur[j] = ceil[i + d] + cnt[j * (h + 1) + i]
            for j in range(n):
                pre[j] = cur[j]
            ceil[i] = max(pre)
        ac.st(ceil[0])
        return

    @staticmethod
    def lg_p1279(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1279
        tag: edit_distance
        """
        s = ac.read_str()
        t = ac.read_str()
        k = ac.read_int()
        m, n = len(s), len(t)

        pre = [j * k for j in range(n + 1)]
        cur = [inf] * (n + 1)
        for i in range(m):
            cur[0] = pre[0] + k
            for j in range(n):
                cur[j + 1] = min(pre[j] + abs(ord(s[i]) - ord(t[j])), cur[j] + k, pre[j + 1] + k)
            for j in range(n + 1):
                pre[j] = cur[j]
        ac.st(pre[-1])
        return

    @staticmethod
    def lg_p1353(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1353
        tag: matrix_dp
        """
        n, m = ac.read_list_ints()

        def idx(i1, j1):
            return i1 * (m + 1) + j1

        nums = [ac.read_int() for _ in range(n)]
        dp = [-inf] * (m + 1) * (n + 1)
        dp[0] = 0
        for i in range(n):
            dp[idx(i + 1, 0)] = dp[idx(i, 0)]
            for j in range(1, ac.min(i + 2, m + 1)):
                dp[idx(i + 1, 0)] = ac.max(dp[idx(i + 1, 0)], dp[idx(i + 1 - j, j)])
            for j in range(1, m + 1):
                dp[idx(i + 1, j)] = dp[idx(i, j - 1)] + nums[i]
        ac.st(dp[idx(n, 0)])
        return

    @staticmethod
    def lg_p1854(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1854
        tag: prefix_max|matrix_dp|specific_plan
        """
        m, n = ac.read_list_ints()
        grid = []
        for _ in range(m):
            grid.extend(ac.read_list_ints())

        def idx(ii, jj):
            return ii * (n + 1) + jj

        dp = [-inf] * (n + 1) * (m + 1)
        for j in range(n + 1):
            dp[j] = 0
        pre = [-1] * (n + 1) * (m + 1)
        for i in range(m):
            x = dp[idx(i, i)]
            ind = i
            for j in range(i, n):
                if dp[idx(i, j)] > x:
                    x = dp[idx(i, j)]
                    ind = j
                if dp[idx(i + 1, j + 1)] < x + grid[i * n + j]:
                    dp[idx(i + 1, j + 1)] = x + grid[i * n + j]
                    pre[idx(i + 1, j + 1)] = ind

        res = max(dp[idx(m, j)] for j in range(n + 1))
        ac.st(res)
        ans = []
        for j in range(n + 1):
            if dp[idx(m, j)] == res:
                ans = [j]
                break
        for i in range(m, 1, -1):
            ans.append(pre[idx(i, ans[-1])])
        ans.reverse()
        ac.lst([x for x in ans])
        return

    @staticmethod
    def lg_p2140(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2140
        tag: matrix_dp
        """
        m, n, u = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        m, n = len(grid), len(grid[0])

        def idx(ii, jj):
            return ii * (n + 1) + jj

        pre = [0] * (n + 1) * (m + 1)
        for i in range(m):
            for j in range(n):
                pre[idx(i + 1, j + 1)] = pre[idx(i, j + 1)] + pre[idx(i + 1, j)] - pre[idx(i, j)] + grid[i][j]

        def idy(ii, jj, kk, pp):
            return ii * n * m * n + jj * m * n + kk * n + pp

        dp = [(-inf, -inf)] * m * n * m * n

        s = pre[-1]
        for xa in range(m - 1, -1, -1):
            for ya in range(n - 1, -1, -1):
                for xb in range(xa, m):
                    for yb in range(ya, n):
                        w = pre[idx(xb + 1, yb + 1)] - pre[idx(xb + 1, ya)] - pre[idx(xa, yb + 1)] + pre[idx(xa, ya)]
                        if w < s - u:
                            continue
                        res = (1, u - (s - w))

                        for xx in range(xa, xb):
                            nex1 = dp[idy(xa, ya, xx, yb)][:]
                            nex2 = dp[idy(xx + 1, ya, xb, yb)][:]
                            nex = (nex1[0] + nex2[0], ac.min(nex1[1], nex2[1]))
                            if nex > res:
                                res = nex[:]
                        for yy in range(ya, yb):
                            nex1 = dp[idy(xa, ya, xb, yy)][:]
                            nex2 = dp[idy(xa, yy + 1, xb, yb)][:]
                            nex = (nex1[0] + nex2[0], ac.min(nex1[1], nex2[1]))
                            if nex > res:
                                res = nex[:]
                        dp[idy(xa, ya, xb, yb)] = res
        ac.lst(dp[idy(0, 0, m - 1, n - 1)])
        return

    @staticmethod
    def lg_p2217(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2217
        tag: matrix_dp
        """
        m, n, k = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        avg = sum(sum(g) for g in grid) / k
        m, n = len(grid), len(grid[0])
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + grid[i][j]

        # @lru_cache(None)
        # def dfs(i, j, x, y, w):
        #     if w == 1:
        #         return (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j]-avg)**2
        #     res = inf
        #     for a in range(i, x):
        #         for up in range(1, w):
        #             res = ac.min(res, dfs(i, j, a, y, up)+dfs(a+1, j, x, y, w-up))
        #
        #     for b in range(j, y):
        #         for up in range(1, w):
        #             res = ac.min(res, dfs(i, j, x, b, up) + dfs(i, b+1, x, y, w - up))
        #     return res
        # ans = (dfs(0, 0, m-1, n-1, k)/k)**0.5
        # ac.st("%.2f" % ans)

        dp = [[[[[inf] * (k + 1) for _ in range(n)] for _ in range(m)] for _ in range(n)] for _ in range(m)]
        for i in range(m - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                for x in range(i, m):
                    for y in range(j, n):
                        for w in range(k + 1):
                            if w == 1:
                                res = (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j] - avg) ** 2
                                dp[i][j][x][y][w] = res
                                continue
                            res = inf
                            for a in range(i, x):
                                for up in range(1, w):
                                    res = ac.min(res, dp[i][j][a][y][up] + dp[a + 1][j][x][y][w - up])
                            for b in range(j, y):
                                for up in range(1, w):
                                    res = ac.min(res, dp[i][j][x][b][up] + dp[i][b + 1][x][y][w - up])
                            dp[i][j][x][y][w] = res
        ans = (dp[0][0][m - 1][n - 1][k] / k) ** 0.5
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def lg_p1436(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1436
        tag: md_matrix_dp
        """
        k = ac.read_int()
        m = n = 8
        grid = [ac.read_list_ints() for _ in range(m)]
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + grid[i][j]

        # @lru_cache(None)
        # def dfs(i, j, x, y, w):
        #     if w == 1:
        #         return (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j])**2
        #     res = inf
        #     for a in range(i, x):
        #         res = ac.min(res, dfs(i, j, a, y, 1)+dfs(a+1, j, x, y, w-1))
        #         res = ac.min(res, dfs(i, j, a, y, w-1) + dfs(a + 1, j, x, y, 1))
        #     for b in range(j, y):
        #         res = ac.min(res, dfs(i, j, x, b, 1) + dfs(i, b+1, x, y, w - 1))
        #         res = ac.min(res, dfs(i, j, x, b, w-1) + dfs(i, b + 1, x, y, 1))
        #     return res
        # ans = dfs(0, 0, m-1, n-1, k)
        # ac.st(ans)

        dp = [[[[[inf] * (k + 1) for _ in range(n)] for _ in range(m)] for _ in range(n)] for _ in range(m)]
        for i in range(m - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                for x in range(i, m):
                    for y in range(j, n):
                        for w in range(k + 1):
                            if w == 1:
                                res = (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j]) ** 2
                                dp[i][j][x][y][w] = res
                                continue
                            res = inf
                            for a in range(i, x):
                                res = ac.min(res, dp[i][j][a][y][1] + dp[a + 1][j][x][y][w - 1])
                                res = ac.min(res, dp[i][j][a][y][w - 1] + dp[a + 1][j][x][y][1])
                            for b in range(j, y):
                                res = ac.min(res, dp[i][j][x][b][1] + dp[i][b + 1][x][y][w - 1])
                                res = ac.min(res, dp[i][j][x][b][w - 1] + dp[i][b + 1][x][y][1])
                            dp[i][j][x][y][w] = res
        ans = dp[0][0][m - 1][n - 1][k]
        ac.st(ans)
        return

    @staticmethod
    def lg_p5752(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5752
        tag: md_matrix_dp
        """
        k = ac.read_int()
        m = n = 8
        grid = [ac.read_list_ints() for _ in range(m)]
        avg = sum(sum(g) for g in grid) / k
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + grid[i][j]

        # @lru_cache(None)
        # def dfs(i, j, x, y, w):
        #     if w == 1:
        #         return (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j]-avg)**2
        #     res = inf
        #     for a in range(i, x):
        #         res = ac.min(res, dfs(i, j, a, y, 1)+dfs(a+1, j, x, y, w-1))
        #         res = ac.min(res, dfs(i, j, a, y, w-1) + dfs(a + 1, j, x, y, 1))
        #     for b in range(j, y):
        #         res = ac.min(res, dfs(i, j, x, b, 1) + dfs(i, b+1, x, y, w - 1))
        #         res = ac.min(res, dfs(i, j, x, b, w-1) + dfs(i, b + 1, x, y, 1))
        #     return res
        # ans = (dfs(0, 0, m-1, n-1, k)/k)**0.5
        # ac.st("%.3f" % ans)

        dp = [[[[[inf] * (k + 1) for _ in range(n)] for _ in range(m)] for _ in range(n)] for _ in range(m)]
        for i in range(m - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                for x in range(i, m):
                    for y in range(j, n):
                        for w in range(k + 1):
                            if w == 1:
                                res = (pre[x + 1][y + 1] - pre[x + 1][j] - pre[i][y + 1] + pre[i][j] - avg) ** 2
                                dp[i][j][x][y][w] = res
                                continue
                            res = inf
                            for a in range(i, x):
                                res = ac.min(res, dp[i][j][a][y][1] + dp[a + 1][j][x][y][w - 1])
                                res = ac.min(res, dp[i][j][a][y][w - 1] + dp[a + 1][j][x][y][1])
                            for b in range(j, y):
                                res = ac.min(res, dp[i][j][x][b][1] + dp[i][b + 1][x][y][w - 1])
                                res = ac.min(res, dp[i][j][x][b][w - 1] + dp[i][b + 1][x][y][1])
                            dp[i][j][x][y][w] = res
        ans = (dp[0][0][m - 1][n - 1][k] / k) ** 0.5
        ac.st("%.3f" % ans)
        return

    @staticmethod
    def lg_p2380(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2380
        tag: md_matrix_dp
        """
        # prefix_sum与matrix_dp
        while True:
            m, n = ac.read_list_ints()
            if m == n == 0:
                break

            grid_west = []
            for _ in range(m):
                lst = ac.read_list_ints()
                grid_west.append(ac.accumulate(lst))

            grid_north = [[0] * (n + 1)]
            for _ in range(m):
                lst = ac.read_list_ints()
                grid_north.append([grid_north[-1][i] + lst[i] for i in range(n)])

            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    # 只能往左或者往上挖
                    dp[i + 1][j + 1] = ac.max(dp[i][j + 1] + grid_west[i][j + 1], dp[i + 1][j] + grid_north[i + 1][j])
            ac.st(dp[-1][-1])
        return

    @staticmethod
    def lg_p2401(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2401
        tag: md_matrix_dp
        """
        # matrix_dp|
        n, k = ac.read_list_ints()
        dp = [[0] * (k + 1) for _ in range(2)]
        pre = 0
        dp[pre][0] = 1
        mod = 2015
        for i in range(n):
            cur = 1 - pre
            dp[cur][0] = 1
            for j in range(1, ac.min(i + 1, k + 1)):
                dp[cur][j] = (dp[pre][j] * (j + 1) + dp[pre][j - 1] * (i - j + 1)) % mod
            pre = cur
        ac.st(dp[pre][-1])
        return

    @staticmethod
    def lg_p2528(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2528
        tag: reverse_order_pair|matrix_dp|implemention|construction
        """
        # reverse_order_pair|matrix_dp| 与implementionconstruction
        n, t = ac.read_list_ints()
        dp = [[0] * (t + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(n):
            dp[i + 1][0] = 1
            for j in range(1, t + 1):
                dp[i + 1][j] = sum(dp[i][j - k] for k in range(min(i, j) + 1))
        ac.st(dp[-1][-1])

        lst = list(range(1, n + 1))
        ans = []
        for _ in range(n):
            m = len(lst)
            for i in range(m):
                rest = (m - 1) * (m - 2) // 2 + i
                if rest >= t:
                    ans.append(lst.pop(i))
                    t -= i
                    break
        ac.lst(ans)
        return

    @staticmethod
    def lg_p2733(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2733
        tag: diff_array|matrix_dp|counter|maximum_square
        """
        # DP通过边长与diff_array|正方形子矩阵的个数
        n = ac.read_int()
        grid = [ac.read_str() for _ in range(n)]
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        diff = [0] * (n + 1)
        for i in range(n):
            for j in range(n):
                if grid[i][j] == "1":
                    # 转移公式
                    dp[i + 1][j + 1] = min(dp[i][j], dp[i + 1][j], dp[i][j + 1]) + 1
                    x = dp[i + 1][j + 1]
                    if x >= 2:
                        diff[2] += 1
                        if x + 1 <= n:
                            diff[x + 1] -= 1
        for i in range(2, n + 1):
            diff[i] += diff[i - 1]
            if diff[i] > 0:
                ac.lst([i, diff[i]])
        return

    @staticmethod
    def lg_p2736(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2736
        tag: matrix_dp
        """
        # matrix_dp|
        n, t, m = ac.read_list_ints()
        nums = ac.read_list_ints()

        # @lru_cache(None)
        # def dfs(i, j, pre):
        #     if i == n:
        #         return 0
        #     if j == m:
        #         return 0
        #     res = dfs(i + 1, j, pre)
        #     if pre + nums[i] <= t:
        #         res = ac.max(res, dfs(i + 1, j, pre + nums[i]) + 1)
        #     if nums[i] <= t and j + 1 < m:
        #         res = ac.max(res, dfs(i + 1, j + 1, nums[i]) + 1)
        #     return res
        #
        # ans = dfs(0, 0, 0)
        # ac.st(ans)

        dp = [[[0] * (t + 1) for _ in range(m + 1)] for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                for k in range(t, -1, -1):
                    res = dp[i + 1][j][k]
                    if k + nums[i] <= t:
                        res = ac.max(res, dp[i + 1][j][k + nums[i]] + 1)
                    if nums[i] <= t and j + 1 < m:
                        res = ac.max(res, dp[i + 1][j + 1][nums[i]] + 1)
                    dp[i][j][k] = res
        ac.st(dp[0][0][0])
        return

    @staticmethod
    def lg_p2769(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2769
        tag: matrix_dp
        """
        # matrix_dp| 注意初始化条件
        n = ac.read_int()
        a = ac.read_list_ints()
        a.sort()
        m = ac.read_int()
        b = ac.read_list_ints()
        b.sort()

        # @lru_cache(None)
        # def dfs(i, j, state):
        #     if i == n:
        #         return 0 if j == m-1 and state else inf
        #     if not state:
        #         return abs(a[i]-b[j])+dfs(i+1, j, 1)
        #
        #     res = dfs(i+1, j, 1)+abs(a[i]-b[j])
        #     if state and j+1<m:
        #         res = min(res, dfs(i+1, j+1, 1)+abs(a[i]-b[j+1]))
        #     return res
        # ac.st(dfs(0, 0, 0))

        dp = [[inf for _ in range(m + 1)] for _ in range(2)]
        pre = 0
        dp[pre][0] = 0
        for i in range(n):
            cur = 1 - pre
            dp[cur][0] = inf
            for j in range(m):
                dp[cur][j + 1] = ac.min(dp[pre][j] + abs(a[i] - b[j]), dp[pre][j + 1] + abs(a[i] - b[j]))
            pre = cur
        ac.st(dp[pre][-1])
        return

    @staticmethod
    def lg_p3012(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3012
        tag: matrix_dp
        """
        # matrix_dp| 可以按照顺序转移
        u, l, p = ac.read_list_ints()
        dct = defaultdict(list)
        nodes = set()
        for _ in range(p):
            st = ac.read_str()
            dct[st[0]].append(st[1])
            nodes.add(st[0])
            nodes.add(st[1])
        nodes = list(nodes)
        ind = {w: i for i, w in enumerate(nodes)}
        m = len(ind)
        mod = 97654321

        # 大写字母个数，小写字母个数，当前结尾字母
        dp = [[[0] * m for _ in range(l + 1)] for _ in range(u + 1)]
        for w in nodes:
            if w.isupper():  # 初始化
                dp[1][0][ind[w]] = 1
            else:
                dp[0][1][ind[w]] = 1

        # 从小到大
        for i in range(u + 1):
            for j in range(l + 1):
                for k in range(m):
                    for nex in dct[nodes[k]]:
                        # 状态转移
                        if nex.isupper() and i + 1 <= u:
                            dp[i + 1][j][ind[nex]] += dp[i][j][k]
                            dp[i + 1][j][ind[nex]] %= mod
                        if nex.islower() and j + 1 <= l:
                            dp[i][j + 1][ind[nex]] += dp[i][j][k]
                            dp[i][j + 1][ind[nex]] %= mod
        ac.st(sum(dp[u][l]) % mod)
        return

    @staticmethod
    def lg_p3860(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3860
        tag: matrix_dp|specific_plan
        """
        # matrix_dp| 并具体转移specific_plan
        n, m = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        pre = [[[0, 0] for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(m):
            dp[i + 1][0] = 0
            for j in range(n):
                cur = post = 0
                for k in range(j, -1, -1):
                    cur += post
                    cur += nums[k]
                    post += nums[k]
                    if cur + dp[i][k] < dp[i + 1][j + 1]:
                        pre[i + 1][j + 1] = [i, k]
                        dp[i + 1][j + 1] = cur + dp[i][k]
        ac.st(dp[m][n])
        ans = [[m, n]]
        while len(ans) < m + 1:
            ans.append(pre[ans[-1][0]][ans[-1][1]])
        ans.reverse()
        x = len(ans)
        for i in range(1, x):
            a, b = ans[i - 1]
            c, d = ans[i]
            ac.st(d - b)
        return

    @staticmethod
    def lg_p4958(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4958
        tag: linear_dp|prefix_sum
        """
        # 三维linear_dpprefix_sum优化
        mod = 10 ** 9 + 7
        ind = {chr(i + ord("a")): i for i in range(26)}
        ind["#"] = 26
        s = ac.read_str()
        n = len(s)
        dp = [[[0] * (n + 1) for _ in range(27)] for _ in range(27)]
        dp[26][26][0] = 1
        pre = [[0] * (n + 1) for _ in range(27)]
        pre[26][0] = 1
        for w in s:
            x = ind[w]
            for k in range(n - 1, -1, -1):
                for j in range(27):
                    dp[x][j][k + 1] += pre[j][k]
                    dp[x][j][k + 1] %= mod
                    pre[x][k + 1] += pre[j][k]
                    pre[x][k + 1] %= mod
        for _ in range(ac.read_int()):
            n, st = ac.read_list_strs()
            n = int(n)
            i, j = ind[st[1]], ind[st[0]]
            ac.st(dp[i][j][n])
        return

    @staticmethod
    def lg_p5144(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5144
        tag: linear_dp|prefix_xor
        """
        # linear_dp 二维|前缀异或和
        n, m = ac.read_list_ints()
        dp = [[0] * m for _ in range(n)]
        nums = ac.read_list_ints()
        dp[0][0] = nums[0]
        for i in range(1, n):
            dp[i][0] = dp[i - 1][0] ^ nums[i]
            for j in range(1, m):
                if j > i:
                    break
                cur = nums[i]
                for k in range(i - 1, -1, -1):
                    dp[i][j] = ac.max(dp[k][j - 1] + cur, dp[i][j])
                    cur ^= nums[k]
        ac.st(dp[-1][-1])
        return

    @staticmethod
    def lg_p5858(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5858
        tag: matrix_dp|monotonic_queue
        """
        # matrix_dp| |monotonic_queue
        n, w, s = ac.read_list_ints()
        nums = ac.read_list_ints()
        dp = [[-inf] * w for _ in range(2)]
        pre = 0
        dp[pre][0] = nums[0]
        for i in range(1, n):
            a = nums[i]
            cur = 1 - pre
            stack = deque()
            x = 0
            for j in range(w):
                if j > i + 1:
                    break
                # monotonic_queue求区间内最大值
                while stack and stack[0][0] < j - 1:
                    stack.popleft()
                while x < w and x <= j + s - 1:
                    while stack and stack[-1][1] <= dp[pre][x]:
                        stack.pop()
                    stack.append([x, dp[pre][x]])
                    x += 1
                if stack:
                    dp[cur][j] = stack[0][1] + (j + 1) * a
            pre = cur
        ac.st(max(dp[pre]))
        return

    @staticmethod
    def lg_p5879(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5879
        tag: matrix_dp|prefix_sum
        """
        # matrix_dp| 后缀和优化
        n = ac.read_int()
        pre = [1] * (n + 1)
        pre[0] = 0
        for x in range(n - 1, 0, -1):
            cur = [0] * (x + 1)
            cnt = pre[-1]
            for j in range(x, -1, -1):
                cnt += pre[j]
                cur[j] = cnt
            pre = cur[:]
        ac.st(sum(pre))
        return

    @staticmethod
    def lg_p6119(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6119
        tag: matrix_dp|lcs
        """
        # matrix_dp| 为 LCS 的变形题
        n = ac.read_int()
        a = [ac.read_int() for _ in range(n)]
        b = [ac.read_int() for _ in range(n)]
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n):
            for j in range(n):
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1], dp[i][j] + int(abs(a[i] - b[j]) <= 4))
        ac.st(dp[-1][-1])
        return

    @staticmethod
    def lg_p6323(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6323
        tag: reverse_order_pair|prefix_sum
        """
        #  DP reverse_order_pair|为指定数量时的排列个数prefix_sum优化
        mod = 10 ** 9 + 7
        n, k = ac.read_list_ints()
        dp = [[0] * (k + 1) for _ in range(2)]
        pre = 0
        dp[pre][0] = 1
        for i in range(n):
            cur = 1 - pre
            lst = ac.accumulate(dp[pre])
            for j in range(k + 1):
                left = j - i if j - i >= 0 else 0
                dp[cur][j] = (lst[j + 1] - lst[left]) % mod
            pre = cur
        ac.st(dp[pre][k] % mod)
        return

    @staticmethod
    def lg_p6394(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6394
        tag: matrix_dp|prefix_sum
        """
        # matrix_dp| |prefix_sum优化
        n, k = ac.read_list_ints()
        s = ac.read_list_ints()
        if sum(s) < n:
            ac.st("impossible")
            return
        mod = 10086001
        dp = [[0] * (n + 1) for _ in range(2)]
        pre = ans = 0
        dp[pre][0] = 1
        for i in range(k):
            cur = 1 - pre
            lst = ac.accumulate(dp[pre])
            for j in range(n + 1):
                low = ac.max(0, j - s[i])
                dp[cur][j] = lst[j + 1] - lst[low]
                dp[cur][j] %= mod
            ans += dp[cur][n]
            ans %= mod
            pre = cur
        ac.st(ans)
        return

    @staticmethod
    def lg_p6433(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6433
        tag: greedy|classification_discussion|matrix_dp
        """
        # greedy|classification_discussionmatrix_dp|
        n, m, k = ac.read_list_ints()

        nums = [ac.read_list_ints() for _ in range(n)]
        if sum(x for _, x in nums) <= m:
            lst = [a for a, _ in nums]
            lst.sort(reverse=True)
            lst.pop()
            ans = sum(a * 2 for a in lst[:k]) + sum(lst[k:])
            ac.st(ans)
            return

        # dp[i][j]表示花费时间 i 翻倍次数为 j 时的最大毒瘤程度
        dp = [[0 for _ in range(k + 1)] for _ in range(m + 1)]
        for a, x in nums:
            for i in range(m, -1, -1):
                for j in range(k, -1, -1):
                    cur = dp[i][j]
                    if i >= x:
                        cur = ac.max(cur, dp[i - x][j] + a)
                        if j >= 1:
                            cur = ac.max(cur, dp[i - x][j - 1] + 2 * a)
                    dp[i][j] = cur
        ac.st(dp[m][k])
        return

    @staticmethod
    def lg_p6451(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6451
        tag: md_matrix_dp|brute_force|4-tree
        """
        # 迭代方式实现md_matrix_dp 并brute_force四叉树获取对应最小代价和状态
        n = ac.read_int()
        grid = [[int(w) for w in ac.read_str()] for _ in range(n)]
        pre = PreFixSumMatrix(grid)
        del grid
        states = list(set([tuple(item) for item in permutations([0, 1, 2, 2], 4)]))
        ind = {state: i for i, state in enumerate(states)}

        def dfs():
            # 最小代价
            stack = [[0, 0, n - 1, n - 1]]
            while stack:
                x1, y1, x2, y2 = stack.pop()
                if x1 >= 0:
                    if (x1, y1, x2, y2) in dct:
                        continue
                    if x1 == x2 and y1 == y2:
                        dct[(x1, y1, x2, y2)] = [0, 0]
                        continue
                    stack.append([~x1, y1, x2, y2])
                    m = (x2 - x1 + 1) // 2
                    x_mid = x1 + m - 1
                    y_mid = y1 + m - 1
                    sub = [[x1, y1, x_mid, y_mid], [x1, y_mid + 1, x_mid, y2],
                           [x_mid + 1, y1, x2, y_mid], [x_mid + 1, y_mid + 1, x2, y2]]
                    stack.extend(sub)
                else:
                    x1 = ~x1
                    m = (x2 - x1 + 1) // 2
                    x_mid = x1 + m - 1
                    y_mid = y1 + m - 1
                    sub = [[x1, y1, x_mid, y_mid], [x1, y_mid + 1, x_mid, y2],
                           [x_mid + 1, y1, x2, y_mid], [x_mid + 1, y_mid + 1, x2, y2]]
                    res = [0, inf]
                    for item in states:
                        cost = 0
                        for i in range(4):
                            xx1, yy1, xx2, yy2 = sub[i]
                            if item[i] == 0:
                                cost += pre.query(xx1, yy1, xx2, yy2)
                            elif item[i] == 1:
                                cost += (yy2 - yy1 + 1) * (xx2 - xx1 + 1) - pre.query(xx1, yy1, xx2, yy2)
                            else:
                                nex = dct[(xx1, yy1, xx2, yy2)]
                                cost += nex[-1]
                        if cost < res[-1]:
                            res = [ind[item], cost]
                    dct[(x1, y1, x2, y2)] = res
            return

        def check():
            # 通过转移状态结果赋值
            stack = [[0, 0, n - 1, n - 1]]
            while stack:
                x1, y1, x2, y2 = stack.pop()
                if x1 == x2 and y1 == y2:
                    ans[x1][y1] = pre.query(x1, y1, x1, y1)
                    continue
                m = (x2 - x1 + 1) // 2
                x_mid = x1 + m - 1
                y_mid = y1 + m - 1
                sub = [[x1, y1, x_mid, y_mid], [x1, y_mid + 1, x_mid, y2],
                       [x_mid + 1, y1, x2, y_mid], [x_mid + 1, y_mid + 1, x2, y2]]
                res = states[dct[(x1, y1, x2, y2)][0]]
                for i in range(4):
                    xx1, yy1, xx2, yy2 = sub[i]
                    if res[i] == 0:
                        continue
                    if res[i] == 1:
                        for w in range(xx1, xx2 + 1):
                            for h in range(yy1, yy2 + 1):
                                ans[w][h] = 1
                    else:
                        stack.append([xx1, yy1, xx2, yy2])
            return

        dct = dict()
        dfs()
        ans = [[0] * n for _ in range(n)]
        check()
        ac.st(dct[(0, 0, n - 1, n - 1)][-1])
        for a in ans:
            ac.st("".join(str(x) for x in a))
        return

    @staticmethod
    def lc_2556(grid: List[List[int]]) -> bool:
        """
        url: https://leetcode.cn/problems/disconnect-path-in-a-binary-matrix-by-at-most-one-flip/description/
        tag: matrix_dp|brain_teaser
        """
        # matrix_dpbrain_teaser|，判断cut_point可行性
        m, n = len(grid), len(grid[0])

        left = [[0] * n for _ in range(m)]
        left[0][0] = 1
        for i in range(m):
            for j in range(n):
                if i == j == 0 or grid[i][j] == 0:
                    continue
                if i - 1 >= 0 and left[i - 1][j]:
                    left[i][j] = 1
                if j - 1 >= 0 and left[i][j - 1]:
                    left[i][j] = 1
        if left[-1][-1] == 0:
            return True

        right = [[0] * n for _ in range(m)]
        right[-1][-1] = 1
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if (i == m - 1 and j == n - 1) or grid[i][j] == 0:
                    continue
                if i + 1 < m and right[i + 1][j]:
                    right[i][j] = 1
                if j + 1 < n and right[i][j + 1]:
                    right[i][j] = 1
        if right[0][0] == 0:
            return True

        dct = defaultdict(int)
        for i in range(m):
            for j in range(n):
                if (i == m - 1 and j == n - 1) or (i == 0 and j == 0):
                    continue
                if left[i][j] and right[i][j]:
                    dct[i + j] += 1
        return True if dct and min(dct.values()) == 1 else False

    @staticmethod
    def lc_2617_1(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/
        tag: reverse_order|matrix_dp|tree_array|prefix_min
        """
        # reverse_order|matrix_dp| 并tree_array|记录更新前缀最小值
        m, n = len(grid), len(grid[0])
        dp = [[inf] * n for _ in range(m)]
        dp[-1][-1] = 1
        row = [PointDescendPreMin(n) for _ in range(m)]
        col = [PointDescendPreMin(m) for _ in range(n)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i == m - 1 and j == n - 1:
                    row[i].point_descend(n, 1)
                    col[j].point_descend(m, 1)
                    continue
                right = grid[i][j] + j + 1 if grid[i][j] + j + 1 < n else n
                val1 = row[i].pre_min(right)

                down = grid[i][j] + i + 1 if grid[i][j] + i + 1 < m else m
                val2 = col[j].pre_min(down)
                dp[i][j] = val1 + 1 if val1 < val2 else val2 + 1
                row[i].point_descend(j + 1, dp[i][j])
                col[j].point_descend(i + 1, dp[i][j])
        return dp[0][0] if dp[0][0] < inf else -1

    @staticmethod
    def lc_2617_2(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/
        tag: reverse_order|matrix_dp|tree_array|prefix_min
        """
        # matrix_dp| priority_queue或者monotonic_queue
        m, n = len(grid), len(grid[0])
        dp = [[inf] * n for _ in range(m)]
        dp[0][0] = 1
        row = [[] for _ in range(m)]
        col = [[] for _ in range(n)]
        heapq.heappush(row[0], [1, grid[0][0]])
        heapq.heappush(col[0], [1, grid[0][0]])
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                while row[i] and row[i][0][1] < j:
                    heapq.heappop(row[i])
                while col[j] and col[j][0][1] < i:
                    heapq.heappop(col[j])
                val = inf if not row[i] else row[i][0][0]
                val = val if not col[j] or col[j][0][0] > val else col[j][0][0]
                dp[i][j] = val + 1
                heapq.heappush(row[i], [val + 1, grid[i][j] + j])
                heapq.heappush(col[j], [val + 1, grid[i][j] + i])
        return dp[-1][-1] if dp[-1][-1] < inf else -1

    @staticmethod
    def lc_2617_3(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/
        tag: reverse_order|matrix_dp|tree_array|prefix_min
        """
        # matrix_dp|  bfs |union_find的方式
        m, n = len(grid), len(grid[0])
        row = [list(range(1, n + 1)) for _ in range(m)]
        col = [list(range(1, m + 1)) for _ in range(n)]
        dp = [[inf] * n for _ in range(m)]
        dp[0][0] = 1
        stack = deque([[0, 0]])
        while stack:
            i, j = stack.popleft()
            d = dp[i][j]
            if i == m - 1 and j == n - 1:
                return d
            val = grid[i][j]

            # union_find或者类似linked_list|合并
            lst = [j]
            # 查到下一个就可以移动到的未访问格子
            while lst[-1] <= j + val and lst[-1] < n:
                lst.append(row[i][lst[-1]])
            last = lst[-1]
            for x in lst[1:-1]:
                if dp[i][x] == inf:
                    dp[i][x] = d + 1
                    stack.append([i, x])
                row[i][x] = last
            row[i][j] = last

            # union_find或者类似linked_list|合并
            lst = [i]
            while lst[-1] <= i + val and lst[-1] < m:
                lst.append(col[j][lst[-1]])
            last = lst[-1]
            for x in lst[1:-1]:
                if dp[x][j] == inf:
                    dp[x][j] = d + 1
                    stack.append([x, j])
                col[j][x] = last
            col[j][i] = last

        return -1

    @staticmethod
    def lg_p6509(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6509
        tag: classical|matrix_dp|specific_plan
        """
        # classicalmatrix_dp| 并记录对应的状态转移
        s = ac.read_str().split("=")
        b = int(s[1])
        s = s[0]
        n = len(s)
        dp = [[inf] * (b + 1) for _ in range(n + 1)]
        dp[0][0] = -1
        pre = [0] * n
        ind = 0
        for i in range(n):
            pre[i] = ind
            if s[i] != "0":
                ind = i
        change = [[[-1, -1] for _ in range(b + 1)] for _ in range(n + 1)]
        for i in range(n):
            j = -1
            for j in range(i, ac.max(-1, i - 5), -1):
                val = int(s[j:i + 1])
                for x in range(b + 1 - val):
                    if dp[j][x] + 1 < dp[i + 1][x + val]:
                        dp[i + 1][x + val] = dp[j][x] + 1
                        change[i + 1][x + val] = [j, x]
            if pre[j] < i - 5:
                j = pre[j] + 1
                val = int(s[j: i + 1])
                for x in range(b + 1 - val):
                    if dp[j][x] + 1 < dp[i + 1][x + val]:
                        dp[i + 1][x + val] = dp[j][x] + 1
                        change[i + 1][x + val] = [j, x]
        ans = list(s)
        x, val = n, b
        while [x, val] != [0, 0]:
            x, val = change[x][val]
            if x:
                ans[x] = "+" + ans[x]
        ac.st("".join(ans) + "=" + str(b))
        return

    @staticmethod
    def lg_p6870(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6870
        tag: matrix_dp|comb|counter
        """
        # matrix_dp| 与组合数优化counter
        n = ac.read_int()
        mod = 10 ** 9 + 7
        cb = Combinatorics(n, mod)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(n + 1):
                for k in range(n + 1 - j):
                    if k == i:
                        continue
                    dp[i][j + k] += dp[i - 1][j] * cb.comb(j + k, k)
                    dp[i][j + k] %= mod

        ans = 1
        for _ in range(n):
            ans *= n
            ans %= mod
        ac.st((ans - dp[n][n]) % mod)
        return

    @staticmethod
    def ac_4418(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4421/
        tag: monotonic_queue|matrix_dp
        """
        # monotonic_queuematrix_dp
        n, k, x = ac.read_list_ints()
        nums = ac.read_list_ints()
        # dp[i][j]表示选第i个元素，且选了j个元素的最大和
        dp = [[-inf] * (x + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        stack = [deque() for _ in range(x + 1)]
        stack[0].append((0, 0))
        for i in range(1, n + 1):
            for j in range(x, 0, -1):
                while stack[j - 1] and stack[j - 1][0][0] < i - k:
                    stack[j - 1].popleft()
                if stack[j - 1]:
                    dp[i][j] = stack[j - 1][0][1] + nums[i - 1]
                while stack[j] and stack[j][-1][1] <= dp[i][j]:
                    stack[j].pop()
                stack[j].append((i, dp[i][j]))

        ans = max(dp[i][x] for i in range(n - k + 1, n + 1))
        ac.st(ans if ans > -inf else -1)
        return

    @staticmethod
    def lc_1216(s: str, k: int) -> bool:
        """
        url: https://leetcode.cn/problems/valid-palindrome-iii/
        tag: matrix_dp|longest_palindrome_subsequence
        """
        # DP求最长回文子序列
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            if i + 1 < n:
                dp[i][i + 1] = 2 if s[i] == s[i + 1] else 1
            for j in range(i + 2, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                a, b = dp[i + 1][j], dp[i][j - 1]
                a = a if a > b else b
                if a > dp[i][j]:
                    dp[i][j] = a
        return n - dp[0][n - 1] <= k

    @staticmethod
    def lg_p7995(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7995
        tag: matrix_dp
        """
        # matrix_dp|
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            k += 1
            grid = [ac.read_str() for _ in range(n)]
            dp = [[[[0, 0, 0] for _ in range(k + 1)] for _ in range(n)] for _ in range(n)]
            dp[0][0][0] = [1, 0, 0]
            for i in range(n):
                for j in range(n):
                    if grid[i][j] == "H":
                        continue
                    if i:
                        d = 1
                        for x in range(k + 1):
                            for y in range(3):
                                kk = x + int(y != d)
                                if kk <= k:
                                    dp[i][j][kk][d] += dp[i - 1][j][x][y]
                    if j:
                        d = 2
                        for x in range(k + 1):
                            for y in range(3):
                                kk = x + int(y != d)
                                if kk <= k:
                                    dp[i][j][kk][d] += dp[i][j - 1][x][y]
            ans = 0
            for x in range(k + 1):
                ans += sum(dp[-1][-1][x])
            ac.st(ans)

        return

    @staticmethod
    def lg_p8325(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8325
        tag: brute_force|matrix_dp|maximum_square
        """
        # 动态规划brute_force，类似最大正方形matrix_dp| 变形
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]

        up = [[0] * n for _ in range(m)]
        for i in range(1, m):
            pre = [-1] * n
            post = [-1] * n
            ind = -1
            for j in range(n):
                pre[j] = ind
                if grid[i][j] == "#":
                    ind = j

            ind = -1
            for j in range(n - 1, -1, -1):
                post[j] = ind
                if grid[i][j] == "#":
                    ind = j

            for j in range(n):
                if grid[i][j] == "." and pre[j] != -1 and post[j] != -1:
                    left = j - pre[j]
                    right = post[j] - j
                    if left == right > 1 and up[i - 1][j] == right - 1:
                        up[i][j] = right
                    if left == right == 1 and grid[i - 1][j] == "#":
                        up[i][j] = 1
        ans = 0
        down = [[0] * n for _ in range(m)]
        for i in range(m - 2, -1, -1):
            pre = [-1] * n
            post = [-1] * n
            ind = -1
            for j in range(n):
                pre[j] = ind
                if grid[i][j] == "#":
                    ind = j

            ind = -1
            for j in range(n - 1, -1, -1):
                post[j] = ind
                if grid[i][j] == "#":
                    ind = j

            for j in range(n):
                if grid[i][j] == "." and pre[j] != -1 and post[j] != -1:
                    left = j - pre[j]
                    right = post[j] - j
                    if left == right > 1 and down[i + 1][j] == right - 1 and right >= 2:
                        down[i][j] = right
                    if left == right == 1 and grid[i + 1][j] == "#":
                        down[i][j] = 1
                if up[i][j] == down[i][j] > 0:
                    ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p8614(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8614
        tag: matrix_dp|mod
        """
        # matrix_dp| 关键在于mod|作为一维状态
        n, s, a, b = ac.read_list_ints()
        mod = 100000007
        dp = [[0] * n for _ in range(n)]
        pre = 0
        dp[pre][0] = 1
        for i in range(1, n):
            cur = 1 - pre
            for j in range(n):
                dp[cur][j] = dp[pre][(j - i * a) % n] + dp[pre][(j + i * b) % n]
                dp[cur][j] %= mod
            pre = cur
        ac.st(dp[pre][s % n])
        return

    @staticmethod
    def lg_p8638(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8638
        tag: matrix_dp|longest_palindrome_sequence
        """
        # matrix_dp| 最长回文子序列
        s = ac.read_str()
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i + 1, n):
                dp[i][j] = ac.max(dp[i + 1][j], dp[i][j - 1])
                if s[i] == s[j] and dp[i + 1][j - 1] + 2 > dp[i][j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
        ac.st(n - dp[0][n - 1])
        return

    @staticmethod
    def lg_p8786(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8786
        tag: classical|md_matrix_dp| implemention|memory_search
        """

        # classical三维matrix_dp| implementionmemory_search

        @lru_cache(None)
        def dfs(x, y, wine):
            if x == 0:
                return 1 if y == wine else 0
            if y == 0 or wine < 0:
                return 0

            res = 0
            if wine * 2 <= y:
                res += dfs(x - 1, y, wine * 2)
            if wine:
                res += dfs(x, y - 1, wine - 1)
            return res % mod

        mod = 10 ** 9 + 7
        n, m = ac.read_list_ints()
        ans = dfs(n, m, 2)
        ac.st(ans)
        return

    @staticmethod
    def lc_2088(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/count-fertile-pyramids-in-a-land/
        tag: matrix_dp
        """

        # 类似求正方形的边长和面积matrix_dp
        def check():
            nonlocal ans
            dp = [[0] * n for _ in range(m)]
            for i in range(m):
                for j in range(n):
                    if grid[i][j]:
                        pre = []
                        for x, y in [[i - 1, j - 1], [i - 1, j], [i - 1, j + 1]]:
                            if 0 <= x < m and 0 <= y < n:
                                pre.append(dp[x][y])
                            else:
                                pre.append(0)
                        dp[i][j] = min(pre) + 1
                        ans += dp[i][j] - 1
            return

        m, n = len(grid), len(grid[0])
        ans = 0
        check()
        grid = grid[::-1]
        check()
        return ans

    @staticmethod
    def lc_2430(s: str) -> int:
        """
        url: https://leetcode.cn/problems/maximum-deletions-on-a-string/
        tag: lcp|matrix_dp
        """
        # 双重DPLCP与matrix_dp
        n = len(s)
        lcp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    lcp[i][j] = lcp[i + 1][j + 1] + 1

        dp = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            for j in range(1, (n - i) // 2 + 1):
                if lcp[i][i + j] >= j:
                    dp[i] = dp[i] if dp[i] > dp[i + j] + 1 else dp[i + j] + 1
        return dp[0]

    @staticmethod
    def ac_4378(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4381/
        tag: classical|matrix_dp
        """
        # classicalmatrix_dp
        n, m, k = ac.read_list_ints()
        dp = [[-inf] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        nums = ac.read_list_ints()
        pre = ac.accumulate(nums)
        if m == 1:
            nums.sort()
            ac.st(sum(nums[-k:]))
            return
        for i in range(n):
            dp[i + 1][0] = 0
            if i >= m - 1:
                for j in range(1, k + 1):
                    a, b = dp[i][j], dp[i - m + 1][j - 1] + \
                                     pre[i + 1] - pre[i - m + 1]
                    dp[i + 1][j] = a if a > b else b
        ac.st(dp[n][k])
        return

    @staticmethod
    def abc_130e(ac=FastIO()):
        # matrix_prefix_sum|优化matrix_dp
        m, n = ac.read_list_ints()
        mod = 10 ** 9 + 7
        s = ac.read_list_ints()
        t = ac.read_list_ints()
        dp = [[0] * n for _ in range(m)]
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if s[i] == t[j]:
                    dp[i][j] = (pre[i][j] + 1) % mod
            for j in range(n):
                pre[i + 1][j + 1] = (pre[i + 1][j] + pre[i][j + 1] - pre[i][j] + dp[i][j]) % mod
        ans = sum(sum(d) for d in dp) + 1
        ac.st(ans % mod)
        return

    @staticmethod
    def ac_2694(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/2696/
        tag: lcs|matrix_dp|counter
        """
        # 问题求解最长公共子序列LCS的长度与个数
        a = ac.read_str()[:-1]
        b = ac.read_str()[:-1]
        mod = 10 ** 8

        # 滚动数组优化
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(2)]
        cnt = [[0] * (n + 1) for _ in range(2)]
        t = 0
        for i in range(n + 1):
            cnt[0][i] = 1
        cnt[1][0] = 1

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cnt[t ^ 1][j] = 0
                if a[i - 1] == b[j - 1]:
                    dp[t ^ 1][j] = dp[t][j - 1] + 1
                    cnt[t ^ 1][j] += cnt[t][j - 1]
                else:
                    dp[t ^ 1][j] = ac.max(dp[t][j], dp[t ^ 1][j - 1])
                # 注意个数去重
                if dp[t ^ 1][j] == dp[t ^ 1][j - 1]:
                    cnt[t ^ 1][j] += cnt[t ^ 1][j - 1]
                if dp[t ^ 1][j] == dp[t][j]:
                    cnt[t ^ 1][j] += cnt[t][j]
                if a[i - 1] != b[j - 1] and dp[t ^ 1][j] == dp[t][j - 1]:
                    cnt[t ^ 1][j] -= cnt[t][j - 1]
                cnt[t ^ 1][j] %= mod
            t ^= 1

        ac.st(dp[t][n])
        ac.st((cnt[t][n] + mod) % mod)
        return

    @staticmethod
    def lc_1594(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-non-negative-product-in-a-matrix/
        tag: matrix_dp|maximum_mul|minimum_mul
        """
        # matrix_dp最大与最小乘积转移
        m, n = len(grid), len(grid[0])

        @lru_cache(None)
        def dfs(i, j):
            if i == m - 1 and j == n - 1:
                return [grid[i][j], grid[i][j]]
            low = inf
            high = -inf
            x = grid[i][j]
            for a, b in [[i + 1, j], [i, j + 1]]:
                if a < m and j < n:
                    res = dfs(a, b)
                    for w in res:
                        low = min(low, w * x)
                        high = max(high, w * x)
            return [low, high]

        ans = dfs(0, 0)[1]
        if ans < 0:
            return -1
        return ans % (10 ** 9 + 7)

    @staticmethod
    def lc_1639(words: List[str], target: str) -> int:
        """
        url: https://leetcode.cn/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/
        tag: prefix_sum|matrix_dp
        """
        # prefix_sum优化matrix_dp|
        dct = defaultdict(lambda: defaultdict(int))
        n = len(words[0])
        for word in words:
            for i, w in enumerate(word):
                dct[w][i] += 1

        m = len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        mod = 10 ** 9 + 7
        for i in range(m):
            dp[i + 1][0] = 0
            pre = dp[i][0]
            for j in range(n):
                c = dct[target[i]][j]
                dp[i + 1][j + 1] = (pre * c) % mod
                pre += dp[i][j + 1]
        return sum(dp[-1]) % mod

    @staticmethod
    def lc_1745(s: str) -> bool:
        """
        url: https://leetcode.cn/problems/palindrome-partitioning-iv/description/
        tag: matrix_dp|palindrome_substring|manacher|brute_force
        """
        # matrix_dp判断是否为palindrome_substring，或者manacher然后brute_force
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            if i + 1 < n:
                dp[i][i + 1] = int(s[i] == s[i + 1])
            for j in range(i + 2, n):
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = 1

        for i in range(1, n - 1):
            for j in range(i, n - 1):
                if dp[i][j] and dp[0][i - 1] and dp[j + 1][n - 1]:
                    return True
        return False

    @staticmethod
    def lc_1771(word1: str, word2: str) -> int:
        """
        url: https://leetcode.cn/problems/maximize-palindrome-length-from-subsequences/
        tag: longest_palindrome_subsequence|matrix_dp
        """
        # 最长回文子序列matrix_dp
        m, n = len(word1), len(word2)
        s = word1 + word2
        ans = 0
        dp = [[0] * (m + n) for _ in range(m + n)]
        for i in range(m + n - 1, -1, -1):
            dp[i][i] = 1
            if i + 1 < m + n:
                dp[i][i + 1] = 2 if s[i] == s[i + 1] else 1
            for j in range(i + 2, m + n):
                a, b = dp[i + 1][j], dp[i][j - 1]
                dp[i][j] = a if a > b else b
                if s[i] == s[j]:
                    a, b = dp[i][j], dp[i + 1][j - 1] + 2
                    dp[i][j] = a if a > b else b
        for i in range(m):
            for j in range(m + n - 1, m - 1, -1):
                if s[i] == s[j]:
                    a, b = ans, dp[i + 1][j - 1] + 2
                    ans = a if a > b else b
                    break
        return ans

    @staticmethod
    def lc_1937(points: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-number-of-points-with-cost/
        tag: prefix_sum|matrix_dp
        """
        # 矩阵prefix_sum后缀和优化的DP
        m, n = len(points), len(points[0])
        pre = points[0][:]

        for i in range(1, m):
            left = [0] * n
            for j in range(n):
                a = -inf if not j else left[j - 1]
                b = pre[j] + j
                left[j] = a if a > b else b

            right = [0] * n
            for j in range(n - 1, -1, -1):
                a = -inf if j == n - 1 else right[j + 1]
                b = pre[j] - j
                right[j] = a if a > b else b

            for j in range(n):
                a = left[j] - j + points[i][j]
                b = right[j] + j + points[i][j]
                pre[j] = a if a > b else b

        return max(pre)

    @staticmethod
    def lc_1977(num: str) -> int:
        """
        url: https://leetcode.cn/problems/number-of-ways-to-separate-numbers/
        tag: matrix_dp|lcp|prefix_sum
        """
        # 两个matrix_dp含LCP优化，或者前缀优化DP
        mod = 10 ** 9 + 7
        n = len(num)
        lcp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            lcp[i][i] = n - i
            for j in range(i + 1, n):
                lcp[i][j] = 0 if num[i] != num[j] else lcp[i + 1][j + 1] + 1

        # 以索引 i 结尾且末尾数字长为 j 的specific_plan数
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        dp[0] = [1] * (n + 1)  # 边界条件prefix_sum
        for i in range(1, n + 1):
            # i 从 1 到 n 表示
            for j in range(1, i + 1):
                if num[i - j] == "0":  # 只能是没有前导零的正整数
                    continue
                if i - 2 * j >= 0:
                    x = lcp[i - 2 * j][i - j]
                    if x >= j or num[i - 2 * j + x] <= num[i - j + x]:
                        dp[i][j] = dp[i - j][j]  # 只有这时才满足 num[i-2*j:i-j] <= num[i-j:i]
                    else:
                        dp[i][j] = dp[i - j][j - 1]
                else:
                    dp[i][j] = dp[i - j][j - 1]
            for j in range(1, n + 1):
                # prefix_sum优化
                dp[i][j] += dp[i][j - 1]
                dp[i][j] %= mod
        return dp[n][n]

    @staticmethod
    def lc_2060(s1: str, s2: str) -> bool:
        """
        url: https://leetcode.cn/problems/check-if-an-original-string-exists-given-two-encoded-strings/description/
        tag: matrix_dp|brute_force|memory_search
        """

        # 二维matrix_dpbrute_forcememory_search

        def check(st):
            if len(st) == 1:
                return [int(st)]
            if len(st) == 2:
                return [int(st), int(st[0]) + int(st[1])]
            return [int(st), int(st[:2]) + int(st[2]), int(st[0]) + int(st[1:]), int(st[0]) + int(st[1]) + int(st[2])]

        def depart(s):
            k = len(s)
            i = 0
            res = []
            while i < k:
                if s[i].isnumeric():
                    cur = ""
                    while i < k and s[i].isnumeric():
                        cur += s[i]
                        i += 1
                    res.append([str(x) for x in check(cur)])
                else:
                    res.append([s[i]])
                    i += 1
            post = []
            for ls in res:
                post.append(max(int(w) if w.isnumeric() else 1 for w in ls))
            return res, list(accumulate(post, initial=0))

        lst1, pre1 = depart(s1)
        lst2, pre2 = depart(s2)
        m, n = len(lst1), len(lst2)

        @lru_cache(None)
        def dfs(i, j, x):
            if pre2[-1] - pre2[j] < x:
                return False
            if pre1[-1] - pre1[i] < -x:
                return False

            if x == 0:
                if i == m and j == n:
                    return True
                if i == m or j == n:
                    return False
                for a in lst1[i]:
                    for b in lst2[j]:
                        if a.isnumeric() and b.isnumeric():
                            if dfs(i + 1, j + 1, int(a) - int(b)):
                                return True
                        elif not a.isnumeric() and not b.isnumeric():
                            if a == b and dfs(i + 1, j + 1, 0):
                                return True
                        elif a.isnumeric() and not b.isnumeric():
                            if dfs(i + 1, j + 1, int(a) - 1):
                                return True
                        else:
                            if dfs(i + 1, j + 1, 1 - int(b)):
                                return True
                return False

            elif x > 0:
                if j == n:
                    return False
                for b in lst2[j]:
                    if b.isnumeric() and dfs(i, j + 1, x - int(b)):
                        return True
                    if not b.isnumeric() and dfs(i, j + 1, x - 1):
                        return True
            else:
                if i == m:
                    return False
                for a in lst1[i]:
                    if a.isnumeric() and dfs(i + 1, j, x + int(a)):
                        return True
                    if not a.isnumeric() and dfs(i + 1, j, x + 1):
                        return True
            return False

        return dfs(0, 0, 0)

    @staticmethod
    def cf_1003f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1003/problem/F
        tag: con_lcp|matrix_dp|string_hash|brute_force
        """
        n = ac.read_int()
        words = ac.read_list_strs()
        lst = [len(w) for w in words]
        eq = [0] * n * n
        for i in range(n):
            eq[i * n + i] = 1
            for j in range(i + 1, n):
                if words[i] == words[j]:
                    eq[i * n + j] = eq[j * n + i] = 1

        dp = [0] * (n + 1) * (n + 1)
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if eq[i * n + j]:
                    dp[i * (n + 1) + j] = dp[(i + 1) * (n + 1) + j + 1] + 1

        ans = sum(len(word) for word in words) + n - 1
        for i in range(n):
            for j in range(i, n):
                x = cnt = 0
                cur = -1
                length = j - i + 1
                while x < n:
                    if dp[i * (n + 1) + x] >= length:
                        cur += 1 + length
                        x += length
                        cnt += 1
                    else:
                        cur += 1 + lst[x]
                        x += 1
                if cnt > 1:
                    ans = ac.min(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lc_2809(nums1: List[int], nums2: List[int], x: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-time-to-make-array-sum-at-most-x/
        tag: matrix_dp|greedy|implemention
        """
        n = len(nums2)
        ind = list(range(n))
        ind.sort(key=lambda it: nums2[it])

        dp = [[0] * (n + 1) for _ in range(2)]
        pre = 0
        for i in range(n):
            cur = 1 - pre
            for j in range(1, i + 2):
                dp[cur][j] = max(dp[pre][j], dp[pre][j - 1] + nums2[ind[i]] * j + nums1[ind[i]])
            pre = cur
        s1 = sum(nums1)
        s2 = sum(nums2)
        for j in range(n + 1):
            if s1 + s2 * j - dp[pre][j] <= x:
                return j
        return -1

    @staticmethod
    def cf_835d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/835/D
        tag: palindrome|matrix_dp
        """
        s = ac.read_str()
        n = len(s)
        dp = [[0] * n for _ in range(2)]
        ans = [0] * (n + 1)
        pre = 0
        for i in range(n - 1, -1, -1):
            cur = 1 - pre
            for j in range(n):
                dp[cur][j] = 0
            dp[cur][i] = 1
            ans[1] += 1
            if i + 1 < n and s[i] == s[i + 1]:
                dp[cur][i + 1] = 2
                ans[2] += 1
            for j in range(i + 2, n):
                if not dp[pre][j - 1] or s[i] != s[j]:
                    continue
                dp[cur][j] = dp[cur][i + (j - i + 1) // 2 - 1] + 1
                ans[dp[cur][j]] += 1
            pre = cur
        for i in range(n - 1, -1, -1):
            ans[i] += ans[i + 1]
        ac.lst(ans[1:])
        return

    @staticmethod
    def cf_1829g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1829/problem/G
        tag: matrix_dp|classical|inclusion_exclusion
        """
        n = 10 ** 6
        dp = [0] * (n + 1)
        pre = [1]
        dp[1] = 1
        x = 2
        father = [[] for _ in range(n + 1)]
        while x <= n:
            cur = list(range(x, x + len(pre) + 1))
            m = len(cur)
            for i in range(m):
                if cur[i] > n:
                    break
                lst = []
                if i:
                    lst.append(pre[i - 1])
                if i < m - 1:
                    lst.append(pre[i])
                father[cur[i]] = lst
                s = sum(dp[y] for y in lst)
                if len(lst) == 2:
                    lst1 = father[lst[0]]
                    lst2 = father[lst[1]]
                    for x1 in lst1:
                        if x1 in lst2:
                            s -= dp[x1]
                dp[cur[i]] = s + cur[i] ** 2
            x += len(pre) + 1
            pre = cur
        for _ in range(ac.read_int()):
            ac.st(dp[ac.read_int()])
        return

    @staticmethod
    def cf_1183h(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1183/problem/H
        tag: matrix_dp|classical|hard|different_sub_sequence
        """
        n, k = ac.read_list_ints()
        s = ac.read_str()
        pre = [-1] * n
        last = [-1] * 26
        for i in range(n):
            w = s[i]
            x = ord(w) - ord("a")
            pre[i] = last[x]
            last[x] = i

        dp = [[0] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(n):
            dp[i + 1][0] = dp[i][0]
            for j in range(1, i + 2):
                dp[i + 1][j] = dp[i][j] + dp[i][j - 1]
                if pre[i] != -1:
                    dp[i + 1][j] -= dp[pre[i]][j - 1]
        ans = 0
        for j in range(n, -1, -1):
            x = ac.min(k, dp[n][j])
            ans += x * (n - j)
            k -= x
        ac.st(ans if not k else -1)
        return

    @staticmethod
    def cf_1353f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1353/problem/F
        tag: matrix_dp|greedy|monotonic_stack
        """
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            grid = [ac.read_list_ints() for _ in range(m)]
            dp = [[] for _ in range(n)]
            tot = m + n - 1
            for i in range(m - 1, -1, -1):
                ndp = [[] for _ in range(n)]
                for j in range(n - 1, -1, -1):
                    if i == m - 1 and j == n - 1:
                        ndp[j] = [(grid[i][j], grid[i][j])]
                        continue
                    lst = []
                    if i + 1 < m:
                        for s, p in dp[j]:
                            cur_s = s + grid[i][j]
                            cur_p = min(p - 1, grid[i][j])
                            lst.append((cur_s, cur_p))
                    if j + 1 < n:
                        for s, p in ndp[j + 1]:
                            cur_s = s + grid[i][j]
                            cur_p = min(p - 1, grid[i][j])
                            lst.append((cur_s, cur_p))
                    lst.sort()
                    cur = []
                    for s, p in lst:
                        if not cur or cur[-1][-1] < p:
                            cur.append((s, p))
                    ndp[j] = cur
                dp = ndp
            ans = 10 ** 18
            for s, p in dp[0]:
                ans = min(ans, s - (p + p + tot - 1) * tot // 2)
            ac.st(ans)
        return

    @staticmethod
    def cf_1593f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1593/problem/F
        tag: matrix_dp|specific_plan|md_vector|flatten
        """

        def get(iii, jjj, ppp, qqq):
            return iii * a * b * (n + 1) + jjj * a * (n + 1) + ppp * (n + 1) + qqq

        for _ in range(ac.read_int()):
            n, a, b = ac.read_list_ints()
            lst = [int(w) for w in ac.read_str()]
            dp = [0] * (n + 1) * a * b * (n + 1)
            dp[get(0, 0, 0, 0)] = 1
            nex_a = [[0] * 10 for _ in range(a)]
            for i in range(a):
                for j in range(10):
                    nex_a[i][j] = (i * 10 + j) % a
            nex_b = [[0] * 10 for _ in range(b)]
            for i in range(b):
                for j in range(10):
                    nex_b[i][j] = (i * 10 + j) % b

            for i in range(n):
                x = lst[i]
                for j in range(b):
                    for p in range(a):
                        for q in range(n + 1):
                            if dp[get(i, j, p, q)]:
                                dp[get(i + 1, j, nex_a[p][x], q + 1)] = 1
                                dp[get(i + 1, nex_b[j][x], p, q)] = 1
            ans = inf
            qq = -1
            for i in range(1, n):
                if dp[get(n, 0, 0, i)]:
                    if abs(n - 2 * i) < ans:
                        ans = abs(n - 2 * i)
                        qq = i
            if ans == inf:
                ac.st(-1)
                continue

            ans = []
            jj = pp = 0
            for i in range(n - 1, -1, -1):
                flag = 0
                x = lst[i]
                for j in range(b):
                    if flag:
                        break
                    for p in range(a):
                        if flag:
                            break
                        for q in range(n + 1):
                            if dp[get(i, j, p, q)]:
                                if (nex_b[j][x], p, q) == (jj, pp, qq):
                                    ans.append("B")
                                    jj, pp, qq = j, p, q
                                    flag = 1
                                    break
                                if (j, nex_a[p][x], q + 1) == (jj, pp, qq):
                                    ans.append("R")
                                    jj, pp, qq = j, p, q
                                    flag = 1
                                    break

            ans.reverse()
            ac.st("".join(ans))
        return

    @staticmethod
    def abc_325f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc325/tasks/abc325_f
        tag: matrix_dp|brute_force|classical
        """
        n = ac.read_int()
        dis = ac.read_list_ints()
        l1, c1, k1 = ac.read_list_ints()
        l2, c2, k2 = ac.read_list_ints()
        dp = [0] * (k1 + 1)
        for i in range(n):
            d = dis[i]
            ndp = [math.inf] * (k1 + 1)
            for j in range(k1 + 1):
                if dp[j] < math.inf:
                    for x in range(k1 - j + 1):
                        need = max(0, math.ceil((d - x * l1) / l2))
                        ndp[j + x] = min(ndp[j + x], dp[j] + need)
            dp = ndp
        ans = math.inf
        for i in range(k1 + 1):
            if dp[i] <= k2:
                ans = min(ans, i * c1 + dp[i] * c2)
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def abc344f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc344/tasks/abc344_f
        tag: matrix_dp|greedy|brain_teaser|classical
        """
        n = ac.read_int()
        pp = [ac.read_list_ints() for _ in range(n)]
        rr = [ac.read_list_ints() for _ in range(n)]
        dd = [ac.read_list_ints() for _ in range(n - 1)]

        dp = [defaultdict(lambda: (math.inf, math.inf)) for _ in range(n)]
        for i in range(n):
            ndp = [defaultdict(lambda: (math.inf, math.inf)) for _ in range(n)]
            for j in range(n):
                if i == j == 0:
                    ndp[0][pp[0][0]] = (0, 0)
                    dp[0][pp[0][0]] = (0, 0)
                    continue
                cur = defaultdict(lambda: (math.inf, math.inf))
                if i - 1 >= 0:
                    for ceil in dp[j]:
                        step, money = dp[j][ceil]
                        money = -money
                        need = dd[i - 1][j]
                        if need > money:
                            cost = (need - money + ceil - 1) // ceil
                        else:
                            cost = 0
                        cur[max(ceil, pp[i][j])] = min(cur[max(ceil, pp[i][j])],
                                                       (step + cost + 1, -(money + cost * ceil - need)))
                if j - 1 >= 0:
                    for ceil in ndp[j - 1]:
                        step, money = ndp[j - 1][ceil]
                        money = -money
                        need = rr[i][j - 1]
                        if need > money:
                            cost = (need - money + ceil - 1) // ceil
                        else:
                            cost = 0
                        cur[max(ceil, pp[i][j])] = min(cur[max(ceil, pp[i][j])],
                                                       (step + cost + 1, -(money + cost * ceil - need)))
                ndp[j] = cur
            dp = ndp
        ans = math.inf
        for ceil in dp[-1]:
            step, money = dp[-1][ceil]
            ans = min(ans, step)
        ac.st(ans)
        return

    @staticmethod
    def abc_311f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc311/tasks/abc311_f
        tag: matrix_dp|prefix_sum_opt|classical|brain_teaser
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        mod = 998244353

        dp = [0] * (m + 1)
        dp[0] = 1
        for j in range(n):
            ndp = [0] * (m + 1)
            pre = ac.accumulate(dp)
            for i in range(m):
                ndp[m - i] = pre[min(m + 1, m - i + 2)]
                if grid[i][j] == "#":
                    break
            else:
                ndp[0] += pre[2]
            dp = [x % mod for x in ndp]
        ac.st(sum(dp) % mod)
        return

    @staticmethod
    def abc_265e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc265/tasks/abc265_e
        tag: matrix_dp|brain_teaser|classical
        """
        mod = 998244353
        n, m = ac.read_list_ints()
        a, b, c, d, e, f = ac.read_list_ints()
        lst = [(a, b), (c, d), (e, f)]
        obs = [ac.read_list_ints() for _ in range(m)]
        obs = set((x, y) for x, y in obs)
        dp = [0] * (n + 1) * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            ndp = [0] * (n + 1) * (n + 1)
            for p in range(i):
                for q in range(i - p):
                    r = i - 1 - p - q
                    aa = p * a + q * c + r * e
                    bb = p * b + q * d + r * f
                    x, y = lst[0]
                    if (x + aa, y + bb) not in obs:
                        ndp[(p + 1) * (n + 1) + q] = (ndp[(p + 1) * (n + 1) + q] + dp[p * (n + 1) + q]) % mod

                    x, y = lst[1]
                    if (x + aa, y + bb) not in obs:
                        ndp[p * (n + 1) + q + 1] = (ndp[p * (n + 1) + q + 1] + dp[p * (n + 1) + q]) % mod

                    x, y = lst[2]
                    if (x + aa, y + bb) not in obs:
                        ndp[p * (n + 1) + q] = (ndp[p * (n + 1) + q] + dp[p * (n + 1) + q]) % mod
            dp = ndp
        ac.st(sum(dp) % mod)
        return

    @staticmethod
    def abc_262d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc262/tasks/abc262_d
        tag: brute_force|matrix_dp|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        mod = 998244353

        ans = 0
        dp = [[0] * n for _ in range(n + 1)]
        for m in range(1, n + 1):
            for i in range(m + 1):
                for j in range(m):
                    dp[i][j] = 0
            dp[0][0] = 1
            for num in nums:
                for j in range(m, 0, -1):
                    for x in range(m):
                        w = (x + num) % m
                        dp[j][w] += dp[(j - 1)][x]
                        dp[j][w] %= mod
            ans += dp[m][0]
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def abc_253e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc253/tasks/abc253_e
        tag: prefix_sum|matrix_dp|inclusion_exclusion|reverse_thinking|classical
        """
        mod = 998244353
        n, m, k = ac.read_list_ints()
        dp = [0] + [1] * m
        for _ in range(1, n):
            pre = ac.accumulate(dp)
            for i in range(1, m + 1):
                low = max(i - (k - 1), 1)
                high = min(k - 1 + i, m)
                if low <= high:
                    dp[i] = (pre[-1] - (pre[high + 1] - pre[low])) % mod
                else:
                    dp[i] = pre[-1] % mod

        ac.st(sum(dp) % mod)
        return

    @staticmethod
    def lc_1883(dist: List[int], speed: int, hours: int) -> int:
        """
        url: https://leetcode.cn/problems/minimum-skips-to-arrive-at-meeting-on-time/
        tag: matrix_dp|high_precision|bag_dp
        """
        n = len(dist)
        dp = [inf] * (n + 1)
        dp[0] = 0
        s = speed
        for d in dist:
            for i in range(n, 0, -1):
                dp[i] = min(dp[i - 1] + d, s * ((dp[i] + s - 1) // s) + d)
            dp[0] = s * ((dp[0] + s - 1) // s) + d
        for i in range(n + 1):
            if dp[i] <= hours * s:
                return i
        return -1
