"""
Algorithm：matrix_dp|memory_search|lcs|md_matrix_dp
Description：matrix_prefix_sum|sub_matrix_sum|maximum_square|edit_distance|lcs|palindrome_substring

====================================LeetCode====================================
174（https://leetcode.com/problems/dungeon-game/）matrix_dp|reverse_thinking
2478（https://leetcode.com/problems/number-of-beautiful-partitions/）matrix_dp
2463（https://leetcode.com/problems/minimum-total-distance-traveled/）matrix_dp
2435（https://leetcode.com/problems/paths-in-matrix-whose-sum-is-divisible-by-k/）matrix_dp|mod
2088（https://leetcode.com/problems/count-fertile-pyramids-in-a-land/）matrix_dp
221（https://leetcode.com/problems/maximal-square/）maximum_square|classical
72（https://leetcode.com/problems/edit-distance/）matrix_dp
329（https://leetcode.com/problems/longest-increasing-path-in-a-matrix/）matrix_dp
1478（https://leetcode.com/problems/allocate-mailboxes/）matrix_dp|median_greedy|interval_dp|greedy
6363（https://leetcode.com/problems/find-the-string-with-lcp/）greedy|construction|lcp
2328（https://leetcode.com/problems/number-of-increasing-paths-in-a-grid/）matrix_dp|counter
2312（https://leetcode.com/problems/selling-pieces-of-wood/）memory_search|specific_plan
2267（https://leetcode.com/problems/check-if-there-is-a-valid-parentheses-string-path/）memory_search
1092（https://leetcode.com/problems/shortest-common-supersequence/）construction|lcs|construction|specific_plan
1143（https://leetcode.com/problems/longest-common-subsequence/）lis|lcs
1035（https://leetcode.com/problems/uncrossed-lines/）lis|lcs
2617（https://leetcode.com/problems/minimum-number-of-visited-cells-in-a-grid/）reverse_order|matrix_dp|tree_array|prefix_min
1092（https://leetcode.com/problems/shortest-common-supersequence/）lis|lcs|specific_plan
1692（https://leetcode.com/problems/count-ways-to-distribute-candies/）matrix_dp|specific_plan|counter
1771（https://leetcode.com/problems/maximize-palindrome-length-from-subsequences/）longest_palindrome_subsequence|matrix_dp
1883（https://leetcode.com/problems/minimum-skips-to-arrive-at-meeting-on-time/）matrix_dp
1977（https://leetcode.com/problems/number-of-ways-to-separate-numbers/）matrix_dp|lcp|prefix_sum
2430（https://leetcode.com/problems/maximum-deletions-on-a-string/）lcp|matrix_dp
1216（https://leetcode.com/problems/valid-palindrome-iii/）matrix_dp|longest_palindrome_subsequence
2060（https://leetcode.com/problems/check-if-an-original-string-exists-given-two-encoded-strings/description/）matrix_dp|brute_force|memory_search
2556（https://leetcode.com/problems/disconnect-path-in-a-binary-matrix-by-at-most-one-flip/description/）matrix_dp|brain_teaser
920（https://leetcode.com/problems/number-of-music-playlists/）matrix_dp
1594（https://leetcode.com/problems/maximum-non-negative-product-in-a-matrix/）matrix_dp|maximum_mul|minimum_mul
1639（https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/）prefix_sum|matrix_dp
956（https://leetcode.com/problems/tallest-billboard/description/）matrix_dp
1301（https://leetcode.com/contest/biweekly-contest-16/problems/number-of-paths-with-max-score/）matrix_dp|specific_plan|counter
1937（https://leetcode.com/problems/maximum-number-of-points-with-cost/）prefix_sum|matrix_dp
1751（https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended-ii/）matrix_dp|binary_search
1959（https://leetcode.com/problems/minimum-total-space-wasted-with-k-resizing-operations/description/）matrix_dp|prefix_sum
1458（https://leetcode.com/problems/max-dot-product-of-two-subsequences/description/）matrix_dp
1745（https://leetcode.com/problems/palindrome-partitioning-iv/description/）matrix_dp|palindrome_substring|manacher|brute_force

=====================================LuoGu======================================
2701（https://www.luogu.com.cn/problem/P2701）maximum_square|matrix_dp|brute_force|classical|O(n^3)|hollow
2049（https://www.luogu.com.cn/problem/P2049）matrix_dp|counter|mod
2138（https://www.luogu.com.cn/problem/P2138）matrix_dp|lcs
1681（https://www.luogu.com.cn/problem/P1681）maximum_square|matrix_dp
2268（https://www.luogu.com.cn/problem/P2268）matrix_dp|edit_distance
2301（https://www.luogu.com.cn/problem/P2301）matrix_dp
2364（https://www.luogu.com.cn/problem/P2364）lcs|specific_plan|matrix_dp
2543（https://www.luogu.com.cn/problem/P2543）matrix_dp|lcs
2513（https://www.luogu.com.cn/record/list?user=739032&status=12&page=2）matrix_dp|prefix_sum
1434（https://www.luogu.com.cn/problem/P1434）matrix_dp|lis
1140（https://www.luogu.com.cn/problem/P1140）matrix_dp
1057（https://www.luogu.com.cn/problem/P1057）matrix_dp
8825（https://www.luogu.com.cn/problem/P8825）mod|matrix_dp|rolling_update
2758（https://www.luogu.com.cn/problem/P2758）matrix_dp|edit_distance
2803（https://www.luogu.com.cn/problem/P2803）matrix_dp|median_greedy|interval_dp
2946（https://www.luogu.com.cn/problem/P2946）matrix_dp
2427（https://www.luogu.com.cn/problem/P2427）matrix_dp|square
7074（https://www.luogu.com.cn/problem/P7074）matrix_dp
7160（https://www.luogu.com.cn/problem/P7160）matrix_dp|brute_force|counter
7266（https://www.luogu.com.cn/problem/P7266）matrix_dp
3399（https://www.luogu.com.cn/problem/P3399）matrix_dp
2516（https://www.luogu.com.cn/problem/P2516）lcs|matrix_dp
1544（https://www.luogu.com.cn/problem/P1544）matrix_dp
1004（https://www.luogu.com.cn/problem/P1004）matrix_dp
1006（https://www.luogu.com.cn/problem/P1006）matrix_dp
1107（https://www.luogu.com.cn/problem/P1107）matrix_dp|prefix_max
1279（https://www.luogu.com.cn/problem/P1279）edit_distance
1353（https://www.luogu.com.cn/problem/P1353）matrix_dp
1410（https://www.luogu.com.cn/problem/P1410）matrix_dp
1799（https://www.luogu.com.cn/problem/P1799）matrix_dp
1854（https://www.luogu.com.cn/problem/P1854）prefix_max|matrix_dp|specific_plan
2140（https://www.luogu.com.cn/problem/P2140）matrix_dp
2217（https://www.luogu.com.cn/problem/P2217）matrix_dp
1436（https://www.luogu.com.cn/problem/P1436）md_matrix_dp
5752（https://www.luogu.com.cn/problem/P5752）md_matrix_dp
2380（https://www.luogu.com.cn/problem/P2380）md_matrix_dp
2401（https://www.luogu.com.cn/problem/P2401）md_matrix_dp
2528（https://www.luogu.com.cn/problem/P2528）reverse_order_pair|matrix_dp|implemention|construction
2733（https://www.luogu.com.cn/problem/P2733）diff_array|matrix_dp|counter|maximum_square
2736（https://www.luogu.com.cn/problem/P2736）matrix_dp
2769（https://www.luogu.com.cn/problem/P2769）matrix_dp
3012（https://www.luogu.com.cn/problem/P3012）matrix_dp
3860（https://www.luogu.com.cn/problem/P3860）matrix_dp|specific_plan
4958（https://www.luogu.com.cn/problem/P4958）linear_dp|prefix_sum
5144（https://www.luogu.com.cn/problem/P5144）linear_dp|prefix_xor
5858（https://www.luogu.com.cn/problem/P5858）matrix_dp|monotonic_queue
5879（https://www.luogu.com.cn/problem/P5879）matrix_dp|prefix_sum
6119（https://www.luogu.com.cn/problem/P6119）matrix_dp|lcs
6323（https://www.luogu.com.cn/problem/P6323）reverse_order_pair|prefix_sum
6394（https://www.luogu.com.cn/problem/P6394）matrix_dp|prefix_sum
6433（https://www.luogu.com.cn/problem/P6433）greedy|classification_discussion|matrix_dp
6451（https://www.luogu.com.cn/problem/P6451）md_matrix_dp|brute_force|4-tree
6509（https://www.luogu.com.cn/problem/P6509）classical|matrix_dp|specific_plan
6870（https://www.luogu.com.cn/problem/P6870）matrix_dp|comb|counter
7995（https://www.luogu.com.cn/problem/P7995）matrix_dp
8325（https://www.luogu.com.cn/problem/P8325）brute_force|matrix_dp|maximum_square
8614（https://www.luogu.com.cn/problem/P8614）matrix_dp|mod
8638（https://www.luogu.com.cn/problem/P8638）matrix_dp|longest_palindrome_sequence
8786（https://www.luogu.com.cn/problem/P8786）classical|md_matrix_dp| implemention|memory_search

===================================CodeForces===================================
1446B（https://codeforces.com/problemset/problem/1446/B）lcs|matrix_dp
429B（https://codeforces.com/problemset/problem/429/B）matrix_dp
1398D（https://codeforces.com/problemset/problem/1398/D）md_matrix_dp|maximum_mul|maximum_sum
2B（https://codeforces.com/problemset/problem/2/B）matrix_dp
1381B（https://codeforces.com/problemset/problem/1381/B）matrix_dp|monotonic_stack
1393D（https://codeforces.com/problemset/problem/1393/D）matrix_dp
1731D（https://codeforces.com/contest/1731/problem/D）binary_search|maximum_square

====================================AtCoder=====================================
E - Common Subsequence（https://atcoder.jp/contests/abc130/tasks/abc130_e）matrix_prefix_sum|matrix_dp

=====================================AcWing=====================================
4378（https://www.acwing.com/problem/content/4381/）classical|matrix_dp
4418（https://www.acwing.com/problem/content/description/4421/）monotonic_queue|matrix_dp
2694（https://www.acwing.com/problem/content/description/2696/）lcs|matrix_dp|counter


"""

import heapq
from collections import defaultdict, deque
from functools import lru_cache
from itertools import permutations, accumulate
from math import inf
from typing import List

from src.basis.diff_array.template import PreFixSumMatrix
from src.data_structure.tree_array.template import PointDescendPreMin
from src.greedy.longest_increasing_subsequence.template import LcsComputeByLis
from src.mathmatics.comb_perm.template import Combinatorics
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1305(nums1: List[int], nums2: List[int]) -> int:
        # LIS的办法求LCS
        return LcsComputeByLis().longest_common_subsequence(nums1, nums2)

    @staticmethod
    def lc_1143(s1: str, s2: str) -> int:
        # LIS的办法求LCS
        return LcsComputeByLis().longest_common_subsequence(s1, s2)

    @staticmethod
    def lc_920(n: int, goal: int, k: int) -> int:
        # matrix_dp（memory_search深搜刷表法实现）
        mod = 10 ** 9 + 7

        @lru_cache(None)  # 前 i 首播放了 r 首不同的歌
        def dfs(i, r):
            if i == goal:
                return 1 if r == n else 0
            res = 0
            if r + 1 <= n:
                res += dfs(i + 1, r + 1) * (n - r)  # 新歌
            if r > k:
                res += dfs(i + 1, r) * (r - k)  # 老歌
            return res % mod

        return dfs(0, 0)

    @staticmethod
    def lc_956(rods: List[int]) -> int:
        # matrix_dp
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
        # 最长公共子序列，并construction包含两个字符串的最短公共超序列
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
        # 标准matrix_dp| 左上到右下的状态转移
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
    def lc_6363(lcp: List[List[int]]) -> str:
        # 根据 LCP 矩阵生成lexicographical_order最小的符合条件的字符串
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
        # 乘积后缀0最少的个数以及对应的路径
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

        # 考虑 0 的存在影响
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
    def cf_1398d(ac, r, g, b, lst):
        # 三维DP，选取两个不同数组的数乘积，最大总和
        @ac.bootstrap
        def dfs(i, j, k):
            if dp[i][j][k] != -1:
                yield
            res = 0
            if i < r and j < g:
                yield dfs(i + 1, j + 1, k)
                res = ac.max(res, dp[i + 1][j + 1][k] + lst[0][i] * lst[1][j])
            if i < r and k < b:
                yield dfs(i + 1, j, k + 1)
                res = ac.max(res, dp[i + 1][j][k + 1] + lst[0][i] * lst[2][k])
            if j < g and k < b:
                yield dfs(i, j + 1, k + 1)
                res = ac.max(res, dp[i][j + 1][k + 1] + lst[2][k] * lst[1][j])
            dp[i][j][k] = res
            yield

        dp = [[[-1] * (b + 1) for _ in range(g + 1)] for _ in range(r + 1)]
        dfs(0, 0, 0)
        return dp[0][0][0]

    @staticmethod
    def lc_2478(s: str, k: int, min_length: int) -> int:
        mod = 10 ** 9 + 7
        # prefix_sum优化二维matrix_dp
        start = set("2357")
        if s[0] not in start:
            return 0
        n = len(s)
        dp = [[0] * n for _ in range(k)]
        for i in range(n):
            if i + 1 >= min_length and s[i] not in start:
                dp[0][i] = 1

        for j in range(1, k):
            pre = 0
            x = 0
            for i in range(n):
                while x <= i - min_length and s[x]:
                    if s[x] not in start and s[x + 1] in start:
                        pre += dp[j - 1][x]
                        pre %= mod
                    x += 1
                if s[i] not in start:
                    dp[j][i] = pre
        return dp[-1][-1]

    @staticmethod
    def lc_2463(robot, factory):
        # 两个数组pointer移动方向与prefix_sum优化求解
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
        # 最长公共子序列的长度以及个数DP
        s = ac.read_str()[:-1]
        t = ac.read_str()[:-1]
        m, n = len(s), len(t)
        mod = 10 ** 8
        # 滚动数组优化
        dp = [[0] * (n + 1) for _ in range(2)]
        cnt = [[0] * (n + 1) for _ in range(2)]
        pre = 0
        for j in range(n + 1):
            cnt[pre][j] = 1
        for i in range(m):
            cur = 1 - pre
            dp[cur][0] = 0
            cnt[cur][0] = 1
            for j in range(n):
                dp[cur][j + 1] = 0
                cnt[cur][j + 1] = 0
                # 长度更长
                if s[i] == t[j]:
                    dp[cur][j + 1] = dp[pre][j] + 1
                    cnt[cur][j + 1] = cnt[pre][j]

                # 左面的转移
                if dp[cur][j] > dp[cur][j + 1]:
                    dp[cur][j + 1] = dp[cur][j]
                    cnt[cur][j + 1] = cnt[cur][j]
                elif dp[cur][j] == dp[cur][j + 1]:
                    cnt[cur][j + 1] += cnt[cur][j]

                # 上面的转移
                if dp[pre][j + 1] > dp[cur][j + 1]:
                    dp[cur][j + 1] = dp[pre][j + 1]
                    cnt[cur][j + 1] = cnt[pre][j + 1]
                elif dp[pre][j + 1] == dp[cur][j + 1]:
                    cnt[cur][j + 1] += cnt[pre][j + 1]

                # 长度未变则设计重复
                if dp[pre][j] == dp[cur][j + 1]:
                    cnt[cur][j + 1] -= cnt[pre][j]
                cnt[cur][j + 1] %= mod
            pre = cur

        ac.st(dp[pre][-1])
        ac.st(cnt[pre][-1])
        return

    @staticmethod
    def lg_p1544(ac=FastIO()):
        # 三维matrix_dp
        n, k = ac.read_list_ints()
        dp = [[[-inf] * (k + 1) for _ in range(n)] for _ in range(2)]
        nums = []
        while len(nums) < n * (n + 1) // 2:
            nums.extend(ac.read_list_ints())

        pre = 0
        num = nums[0]
        dp[pre][0][0] = num
        dp[pre][0][1] = num * 3
        s = 1
        for i in range(1, n):
            lst = nums[s:s + i + 1]
            s += i + 1
            cur = 1 - pre
            dp[cur] = [[-inf] * (k + 1) for _ in range(n)]
            for j in range(i + 1):
                for p in range(k + 1):
                    if j and p:
                        a = ac.max(dp[pre][j][p], dp[pre][j - 1][p]) + lst[j]
                        b = ac.max(dp[pre][j][p - 1], dp[pre][j - 1][p - 1]) + lst[j] * 3
                        dp[cur][j][p] = ac.max(a, b)
                    elif j:
                        dp[cur][j][p] = ac.max(dp[pre][j][p], dp[pre][j - 1][p]) + lst[j]
                    elif p:
                        dp[cur][j][p] = ac.max(dp[pre][j][p] + lst[j], dp[pre][j][p - 1] + lst[j] * 3)
                    else:
                        dp[cur][j][p] = dp[pre][j][p] + lst[j]
            pre = cur
        ac.st(max(max(d) for d in dp[pre]))
        return

    @staticmethod
    def lg_p1004(ac=FastIO()):
        # 取数四维转三维DP，路径可以有交叠
        n = ac.read_int()
        grid = [[0] * n for _ in range(n)]
        while True:
            lst = ac.read_list_ints()
            if lst == [0, 0, 0]:
                break
            x, y, z = lst
            grid[x - 1][y - 1] = z

        dp = [[[0] * n for _ in range(n)] for _ in range(n)]
        for x1 in range(n - 1, -1, -1):
            for y1 in range(n - 1, -1, -1):
                high = ac.min(n - 1, x1 + y1)
                low = ac.max(0, x1 + y1 - (n - 1))
                for x2 in range(high, low - 1, -1):
                    y2 = x1 + y1 - x2
                    post = 0
                    for a, b in [[x1 + 1, y1], [x1, y1 + 1]]:
                        for c, d in [[x2 + 1, y2], [x2, y2 + 1]]:
                            if 0 <= a < n and 0 <= b < n and 0 <= c < n and 0 <= d < n:
                                post = ac.max(post, dp[a][b][c])
                    dp[x1][y1][x2] = post + grid[x1][y1] + grid[x2][y2]
                    if x1 == x2:
                        dp[x1][y1][x2] -= grid[x1][y1]
        ac.st(dp[0][0][0])
        return

    @staticmethod
    def lg_p1006(ac=FastIO()):
        # 取数四维转三维DP，路径不能有交叠
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]

        dp = [[[0] * m for _ in range(n)] for _ in range(m)]
        for x1 in range(m - 1, -1, -1):
            for y1 in range(n - 1, -1, -1):
                high = ac.min(m - 1, x1 + y1)
                low = ac.max(0, x1 + y1 - (n - 1))
                for x2 in range(high, low - 1, -1):
                    y2 = x1 + y1 - x2
                    post = 0
                    for a, b in [[x1 + 1, y1], [x1, y1 + 1]]:
                        for c, d in [[x2 + 1, y2], [x2, y2 + 1]]:
                            if 0 <= a < m and 0 <= b < n and 0 <= c < m and 0 <= d < n:
                                post = ac.max(post, dp[a][b][c])
                    dp[x1][y1][x2] = post + grid[x1][y1] + grid[x2][y2]
                    if x1 == x2 and y1 == y2 and [x1, y1] not in [[0, 0], [m - 1, n - 1]]:
                        dp[x1][y1][x2] = -inf
        ac.st(dp[0][0][0])
        return

    @staticmethod
    def lg_p1107(ac=FastIO()):
        # matrix_dp|前缀数组最值优化
        n, h, d = ac.read_list_ints()
        cnt = [[0] * (h + 1) for _ in range(n)]
        for i in range(n):
            lst = ac.read_list_ints()
            for j in lst[1:]:
                cnt[i][j] += 1

        ceil = [0] * (h + 1)
        dp = [[0] * n for _ in range(2)]
        pre = 0
        for i in range(h, -1, -1):
            cur = 1 - pre
            for j in range(n):
                dp[cur][j] = dp[pre][j] + cnt[j][i]
                if i + d <= h and ceil[i + d] > dp[pre][j]:
                    dp[cur][j] = ceil[i + d] + cnt[j][i]
            pre = cur
            ceil[i] = max(dp[pre])
        ac.st(ceil[0])
        return

    @staticmethod
    def lg_p1279(ac=FastIO()):
        # edit_distance DP 变形
        s = ac.read_str()
        t = ac.read_str()
        k = ac.read_int()
        m, n = len(s), len(t)
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for j in range(n):
            dp[0][j + 1] = dp[0][j] + k
        for i in range(m):
            dp[i + 1][0] = dp[i][0] + k
            for j in range(n):
                dp[i + 1][j + 1] = min(dp[i][j] + abs(ord(s[i]) - ord(t[j])), dp[i + 1][j] + k, dp[i][j + 1] + k)
        ac.st(dp[-1][-1])
        return

    @staticmethod
    def lg_p1353(ac=FastIO()):
        # matrix_dp
        n, m = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]
        dp = [[-inf] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(n):
            dp[i + 1][0] = dp[i][0]
            for j in range(1, min(i + 2, m + 1)):
                dp[i + 1][0] = ac.max(dp[i + 1][0], dp[i + 1 - j][j])
            for j in range(1, m + 1):
                dp[i + 1][j] = dp[i][j - 1] + nums[i]
        ac.st(dp[n][0])
        return

    @staticmethod
    def lg_p1854(ac=FastIO()):
        # matrix_dp，并输出匹配specific_plan
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        dp = [[-inf] * (n + 1) for _ in range(m + 1)]
        dp[0] = [0] * (n + 1)
        pre = [[-1] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            x = dp[i][i]
            ind = i
            for j in range(i, n):
                if dp[i][j] > x:
                    x = dp[i][j]
                    ind = j
                if dp[i + 1][j + 1] < x + grid[i][j]:
                    dp[i + 1][j + 1] = x + grid[i][j]
                    # 记录上一行转移顺序
                    pre[i + 1][j + 1] = ind

        # reverse_order|输出specific_plan
        res = max(dp[m])
        ac.st(res)
        ans = [dp[m].index(res)]
        for i in range(m, 1, -1):
            ans.append(pre[i][ans[-1]])
        ans.reverse()
        ac.lst([x for x in ans])
        return

    @staticmethod
    def lg_p2140(ac=FastIO()):
        # 矩阵四维DP，可以memory_search与迭代
        m, n, u = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        m, n = len(grid), len(grid[0])
        pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pre[i + 1][j + 1] = pre[i][j + 1] + pre[i + 1][j] - pre[i][j] + grid[i][j]
        s = pre[-1][-1]

        # @lru_cache(None)
        # def dfs(xa, ya, xb, yb):
        #     w = pre[xb + 1][yb + 1] - pre[xb + 1][ya] - pre[xa][yb + 1] + pre[xa][ya]
        #     res = [-inf, -inf]
        #     if w >= s-u:
        #         res = [1, u-(s-w)]
        #     else:
        #         return res
        #
        #     for xx in range(xa, xb):
        #         nex1 = dfs(xa, ya, xx, yb)
        #         nex2 = dfs(xx+1, ya, xb, yb)
        #         nex = [nex1[0]+nex2[0], ac.min(nex1[1], nex2[1])]
        #         if nex > res:
        #             res = nex[:]
        #
        #     for yy in range(ya, yb):
        #         nex1 = dfs(xa, ya, xb, yy)
        #         nex2 = dfs(xa, yy+1, xb, yb)
        #         nex = [nex1[0]+nex2[0], ac.min(nex1[1], nex2[1])]
        #         if nex > res:
        #             res = nex[:]
        #     return res
        # ans = dfs(0, 0, m-1, n-1)
        # ac.lst(ans)

        dp = [[[[[-inf, -inf] for _ in range(n)] for _ in range(m)] for _ in range(n)] for _ in range(m)]
        for xa in range(m - 1, -1, -1):
            for ya in range(n - 1, -1, -1):
                for xb in range(xa, m):
                    for yb in range(ya, n):
                        w = pre[xb + 1][yb + 1] - pre[xb + 1][ya] - pre[xa][yb + 1] + pre[xa][ya]
                        if w < s - u:
                            continue
                        res = [1, u - (s - w)]

                        for xx in range(xa, xb):
                            nex1 = dp[xa][ya][xx][yb]
                            nex2 = dp[xx + 1][ya][xb][yb]
                            nex = [nex1[0] + nex2[0], ac.min(nex1[1], nex2[1])]
                            if nex > res:
                                res = nex[:]
                        for yy in range(ya, yb):
                            nex1 = dp[xa][ya][xb][yy]
                            nex2 = dp[xa][yy + 1][xb][yb]
                            nex = [nex1[0] + nex2[0], ac.min(nex1[1], nex2[1])]
                            if nex > res:
                                res = nex[:]
                        dp[xa][ya][xb][yb] = res
        ac.lst(dp[0][0][m - 1][n - 1])
        return

    @staticmethod
    def lg_p2217(ac=FastIO()):

        # 矩阵四维DP，可以memory_search与迭代
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

        # 矩阵四维DP，可以memory_search与迭代
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

        # 矩阵四维DP，可以memory_search与迭代
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
        # matrix_dp判断是否为palindrome_substring，或者马拉车然后brute_force
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