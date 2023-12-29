"""
Algorithm：bag_dp|group_bag_dp|linear_bag_dp|matrix_bag_dp|limited_bag_dp|fill_table(from past)|refresh_table(update future)|lazy_bag_like|inclusion_exclusion
Description：bag_dp|bin_split|finite|infinite|tree_bag_dp

====================================LeetCode====================================
140（https://leetcode.cn/problems/word-break-ii/）bag_dp|specific_plan
2218（https://leetcode.cn/problems/maximum-value-of-k-coins-from-piles/）group_bag_dp
2585（https://leetcode.cn/problems/number-of-ways-to-earn-points/）bag_dp
2189（https://leetcode.cn/problems/number-of-ways-to-build-house-of-cards/）bag_dp
254（https://leetcode.cn/problems/factor-combinations/）bag_dp|mul
1449（https://leetcode.cn/problems/form-largest-integer-with-digits-that-add-up-to-target/）bag_dp
1049（https://leetcode.cn/problems/last-stone-weight-ii/）bag_dp
2742（https://leetcode.cn/problems/painting-the-walls/description/）bag_dp
2518（https://leetcode.cn/problems/number-of-great-partitions/）bag_dp|counter
1155（https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum/description/）group_bag_dp|fill_table|refresh_table
2902（https://leetcode.cn/problems/count-of-sub-multisets-with-bounded-sum/）monotonic_queue|mod|group_bag_dp|prefix_sum|inclusion_exclusion|lazy_bag_like

=====================================LuoGu======================================
P1048（https://www.luogu.com.cn/problem/P1048）bag_dp|finite
P1049（https://www.luogu.com.cn/problem/P1049）bag_dp
P1776（https://www.luogu.com.cn/problem/P1776）matrix_bag_dp|bin_split|monotonic_queue
P1509（https://www.luogu.com.cn/problem/P1509）matrix_bag_dp
P1799（https://www.luogu.com.cn/problem/P1799）matrix_bag_dp
P1566（https://www.luogu.com.cn/problem/P1566）counter|limited|bag_dp
P1794（https://www.luogu.com.cn/problem/P1794）matrix_bag_dp
P1806（https://www.luogu.com.cn/problem/P1806）bag_dp|counter
P1853（https://www.luogu.com.cn/problem/P1853）bag_dp|infinite
P1874（https://www.luogu.com.cn/problem/P1874）brute_force|bag_dp
P1977（https://www.luogu.com.cn/problem/P1977）group_bag_dp|finite
P1586（https://www.luogu.com.cn/problem/P1586）group_bag_dp|infinite
P1566（https://www.luogu.com.cn/problem/P1566）bag_dp|counter
P1509（https://www.luogu.com.cn/problem/P1509）matrix_bag_dp
P1504（https://www.luogu.com.cn/problem/P1504）bag_dp|finite
P2066（https://www.luogu.com.cn/problem/P2066）group_bag_dp|finite
P2340（https://www.luogu.com.cn/problem/P2340）bag_dp
P2370（https://www.luogu.com.cn/problem/P2370）mst|sort|greedy|bag_dp
P2386（https://www.luogu.com.cn/problem/P2386）bag_dp|counter
P2623（https://www.luogu.com.cn/problem/P2623）bag_dp|finite|bin_split|infinite
P1474（https://www.luogu.com.cn/problem/P1474）bag_dp|infinite|counter
P1466（https://www.luogu.com.cn/problem/P1466）bag_dp|finite|和counter
P1455（https://www.luogu.com.cn/problem/P1455）union_find|bag_dp|finite|
P1230（https://www.luogu.com.cn/problem/P1230）sort|bag_dp|finite|
P1077（https://www.luogu.com.cn/problem/P1077）bag_dp|finite|counter
P2725（https://www.luogu.com.cn/problem/P2725）bag_dp|infinite|counter
P2918（https://www.luogu.com.cn/problem/P2918）bag_dp|infinite|
P3027（https://www.luogu.com.cn/problem/P3027）bag_dp|infinite
P3030（https://www.luogu.com.cn/problem/P3030）brute_force|group_bag_dp|finite|bag_dp
P3040（https://www.luogu.com.cn/problem/P3040）matrix_bag_dp
P4817（https://www.luogu.com.cn/problem/P4817）bag_dp|finite
P5087（https://www.luogu.com.cn/problem/P5087）matrix_bag_dp
P6205（https://www.luogu.com.cn/problem/P6205）bag_dp|infinite
P6389（https://www.luogu.com.cn/problem/P6389）bag_dp|finite
P6567（https://www.luogu.com.cn/problem/P6567）finite|bag_dp|bin_split|classical
P6771（https://www.luogu.com.cn/problem/P6771）sort|bag_dp|finite|bin_split
P2842（https://www.luogu.com.cn/problem/P2842）bag_dp|infinite
P2840（https://www.luogu.com.cn/problem/P2840）bag_dp|infinite
P2834（https://www.luogu.com.cn/problem/P2834）bag_dp|infinite
P1064（https://www.luogu.com.cn/problem/P1064）bag_dp|finite|brute_force|classification_discussion|group_bag_dp
P1156（https://www.luogu.com.cn/problem/P1156）bag_dp|finite
P1273（https://www.luogu.com.cn/problem/P1273）tree|graph|group_bag_dp
P1284（https://www.luogu.com.cn/problem/P1284）brute_force|triangle|math|bag_dp
P1441（https://www.luogu.com.cn/problem/P1441）brute_force|bag_dp
P1537（https://www.luogu.com.cn/problem/P1537）bin_split|bag_dp
P1541（https://www.luogu.com.cn/problem/P1541）brute_force|matrix_dp|fill_table
P1759（https://www.luogu.com.cn/problem/P1759）matrix_bag_dp|lexicographical_order|specific_plan
P1833（https://www.luogu.com.cn/problem/P1833）infinite|bag_dp|monotonic_queue|matrix_bag_dp
P2014（https://www.luogu.com.cn/problem/P2014）dag|tree_bag_dp
P2079（https://www.luogu.com.cn/problem/P2079）rolling_hash|bag_dp
P2170（https://www.luogu.com.cn/problem/P2170）union_find|bag_dp|finite|bin_split
P2214（https://www.luogu.com.cn/problem/P2214）bag_dp|greedy
P2306（https://www.luogu.com.cn/problem/P2306）data_range|counter|finite|bin_split
P2320（https://www.luogu.com.cn/problem/P2320）bin_split|greedy|reverse_thinking
P2737（https://www.luogu.com.cn/problem/P2737）infinite|bag_dp
P2760（https://www.luogu.com.cn/problem/P2760）monotonic_queue|matrix_bag_dp
P2854（https://www.luogu.com.cn/problem/P2854）bag_dp|group_bag_dp|finite
P2938（https://www.luogu.com.cn/problem/P2938）infinite|group_bag_dp
P2979（https://www.luogu.com.cn/problem/P2979）bag_dp|group_bag_dp|finite
P3010（https://www.luogu.com.cn/problem/P3010）bag_dp|heapq
P3423（https://www.luogu.com.cn/problem/P3423）bin_split|matrix_bag_dp|specific_plan
P3983（https://www.luogu.com.cn/problem/P3983）infinite|bag_dp
P5322（https://www.luogu.com.cn/problem/P5322）matrix_dp|group_bag_dp|classical
P5365（https://www.luogu.com.cn/problem/P5365）bag_dp|infinite|brute_force|counter
P5662（https://www.luogu.com.cn/problem/P5662）infinite|bag_dp|greedy
P1417（https://www.luogu.com.cn/problem/P1417）greedy|sort|bag_dp

===================================CodeForces===================================
577B（https://codeforces.com/problemset/problem/577/B）mod|counter|bin_split|bag_dp
543A（https://codeforces.com/problemset/problem/543/A）matrix_bag_dp
148E（https://codeforces.com/problemset/problem/148/E）bag_dp|finite|brute_force
1433F（https://codeforces.com/problemset/problem/1433/F）bag_dp|finite|brute_force
1657D（https://codeforces.com/contest/1657/problem/D）infinite|bag_dp|mul|euler_series|O(nlogn)||binary_search|greedy

====================================AtCoder=====================================
ABC054D（https://atcoder.jp/contests/abc054/tasks/abc054_d）matrix_bag_dp|finite
ABC118D（https://atcoder.jp/contests/abc118/tasks/abc118_d）greedy|bag_dp|specific_plan
ABC145E（https://atcoder.jp/contests/abc145/tasks/abc145_e）brain_teaser|bag_dp|finite|sort|refresh_table

=====================================AcWing=====================================
4（https://www.acwing.com/problem/content/4/）bin_split|matrix_bag_dp
6（https://www.acwing.com/problem/content/description/6/）monotonic_queue|matrix_bag_dp
7（https://www.acwing.com/problem/content/7/）bag_dp|finite|infinite|matrix_bag_dp
8（https://www.acwing.com/problem/content/8/）matrix_bag_dp|finite
9（https://www.acwing.com/problem/content/9/）bag_dp|group_bag_dp|finite
10（https://www.acwing.com/problem/content/10/）tree_bag_dp
11（https://www.acwing.com/problem/content/description/11/）bag_dp|counter|specific_plan
12（https://www.acwing.com/problem/content/12/）bag_dp|specific_plan
4081（https://www.acwing.com/problem/content/4084/）matrix_bag_dp

"""
import bisect
from collections import defaultdict, deque, Counter
from functools import lru_cache
from itertools import combinations
from typing import List

from src.dp.bag_dp.template import BagDP
from src.graph.union_find.template import UnionFind
from src.mathmatics.number_theory.template import NumberTheory
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1433f(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1433/F
        tag: bag_dp|finite|brute_force
        """

        m, n, k = ac.read_list_ints()
        pre = [-inf] * k
        pre[0] = 0
        x = n // 2
        for _ in range(m):
            nums = ac.read_list_ints()
            dp = [[-inf] * k for _ in range(x + 1)]
            dp[0][0] = 0
            for num in nums:
                nex = [ls[:] for ls in dp]
                for i in range(x):
                    for j in range(k):
                        d = (j + num) % k
                        nex[i + 1][d] = ac.max(dp[i][j] + num, nex[i + 1][d])
                dp = [ls[:] for ls in nex]
            tmp = [max(dp[i][j] for i in range(x + 1)) for j in range(k)]

            cur = pre[:]
            for i in range(k):
                for j in range(k):
                    cur[(i + j) % k] = ac.max(cur[(i + j) % k], pre[i] + tmp[j])
            pre = cur[:]

        ac.st(pre[0])
        return

    @staticmethod
    def cf_543a(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/543/A
        tag: matrix_bag_dp
        """

        n, m, b, mod = ac.read_list_ints()
        nums = ac.read_list_ints()
        pre = [[0] * (b + 1) for _ in range(m + 1)]
        pre[0][0] = 1
        for num in nums:
            for i in range(1, m + 1):
                for j in range(num, b + 1):
                    pre[i][j] = (pre[i][j] + pre[i - 1][j - num]) % mod
        ac.st(sum(pre[m]) % mod)
        return

    @staticmethod
    def cf_577b(m, nums):
        """
        url: https://codeforces.com/problemset/problem/577/B
        tag: mod|counter|bin_split|bag_dp
        """

        cnt = [0] * m
        for num in nums:
            cnt[num % m] += 1
        if cnt[0] or max(cnt) >= m:
            return "YES"
        pre = [0] * m
        for i in range(1, m):
            if cnt[i]:
                for x in BagDP().bin_split_1(cnt[i]):
                    cur = pre[:]
                    y = (x * i) % m
                    cur[y] = 1
                    for j in range(m):
                        if pre[j]:
                            cur[(j + y) % m] = 1
                    pre = cur[:]
                if pre[0]:
                    return "YES"
        return "NO"

    @staticmethod
    def lc_2218(piles: List[List[int]], k: int) -> int:
        """
        url: https://leetcode.cn/problems/maximum-value-of-k-coins-from-piles/
        tag: group_bag_dp
        """

        cur = [0] * (k + 1)
        for lst in piles:
            n = len(lst)
            pre = [0] * (n + 1)
            for i in range(n):
                pre[i + 1] = pre[i] + lst[i]

            nex = cur[:]
            for j in range(1, k + 1):
                for x in range(min(n + 1, j + 1)):
                    nex[j] = max(nex[j], cur[j - x] + pre[x])
            cur = nex[:]
        return cur[-1]

    @staticmethod
    def lg_p6567(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6567
        tag: finite|bag_dp|bin_split|classical
        """

        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        target = ac.read_list_ints()
        ceil = max(target)
        dp = [0] * (ceil + 1)
        dp[0] = 1
        for k, a in nums:
            for b in BagDP().bin_split_1(a):
                x = b * k
                for i in range(ceil, x - 1, -1):
                    if dp[i - x]:
                        dp[i] = 1
        for t in target:
            if dp[t]:
                ac.st("Yes")
            else:
                ac.st("No")
        return

    @staticmethod
    def lc_2742_1(cost: List[int], time: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/painting-the-walls/description/
        tag: bag_dp
        """

        @lru_cache(None)
        def dfs(i, pre):
            if pre >= n - i:
                return 0
            if i == n:
                return inf
            res = dfs(i + 1, pre - 1)
            cur = dfs(i + 1, pre + time[i]) + cost[i]
            if cur < res:
                res = cur
            return res

        n = len(cost)
        return dfs(0, 0)

    @staticmethod
    def lc_2742_2(cost: List[int], time: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/painting-the-walls/description/
        tag: bag_dp
        """

        n = len(cost)
        dp = [sum(time)] * (n + 1)
        dp[0] = 0
        for i in range(n):
            c, t = cost[i], time[i]
            for j in range(n, -1, -1):
                s = j - t - 1 if j - t - 1 >= 0 else 0
                if dp[s] + c < dp[j]:
                    dp[j] = dp[s] + c
        return dp[-1]

    @staticmethod
    def lc_2518(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-great-partitions/
        tag: bag_dp|counter
        """

        mod = 10 ** 9 + 7
        dp = [0] * k
        s = sum(nums)
        if s < 2 * k:
            return 0
        dp[0] = 1
        n = len(nums)
        for num in nums:
            for i in range(k - 1, num - 1, -1):
                dp[i] += dp[i - num]
        ans = pow(2, n, mod)
        ans -= 2 * sum(dp)
        return ans % mod

    @staticmethod
    def lc_2585(target: int, types: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/number-of-ways-to-earn-points/
        tag: bag_dp
        """

        mod = 10 ** 9 + 7
        n = len(types)
        pre = [0] * (target + 1)
        pre[0] = 1
        for i in range(n):
            c, m = types[i]
            cur = pre[:]
            for x in range(1, c + 1):
                for j in range(target - x * m + 1):
                    if x * m + j <= target:
                        cur[x * m + j] += pre[j]
            pre = [num % mod for num in cur]
        return pre[-1]

    @staticmethod
    def cf_1657d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1657/problem/D
        tag: infinite|bag_dp|mul|euler_series|O(nlogn)||binary_search|greedy
        """
        n, c = ac.read_list_ints()
        dp = [0] * (c + 1)
        for _ in range(n):
            cc, dd, hh = ac.read_list_ints()
            dp[cc] = ac.max(dp[cc], dd * hh)

        for i in range(1, c + 1):
            dp[i] = ac.max(dp[i], dp[i - 1])
            x = dp[i]
            for y in range(i * 2, c + 1, i):
                dp[y] = ac.max(dp[y], x * (y // i))

        ans = []
        for _ in range(ac.read_int()):
            h, d = ac.read_list_ints()
            if h * d >= dp[c]:
                ans.append(-1)
            else:
                ans.append(bisect.bisect_right(dp, h * d))
        ac.lst(ans)
        return

    @staticmethod
    def lc_254(n: int) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/factor-combinations/
        tag: bag_dp|mul
        """

        lst = NumberTheory().get_all_factor(n)
        m = len(lst)
        dp = defaultdict(list)
        dp[1] = [[]]
        for i in range(1, m - 1):
            for j in range(i, m):
                if lst[j] % lst[i] == 0:
                    x = lst[j] // lst[i]
                    for p in dp[x]:
                        dp[lst[j]].append(p + [lst[i]])
        return [ls for ls in dp[n] if ls]

    @staticmethod
    def abc_118d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc118/tasks/abc118_d
        tag: greedy|bag_dp|specific_plan
        """
        score = [2, 5, 5, 4, 5, 6, 3, 7, 6]
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        nums.sort(reverse=True)
        dp = [-inf] * (n + 1)
        dp[0] = 0
        for num in nums:
            val = score[num - 1]
            for i in range(val, n + 1):
                if dp[i - val] + 1 > dp[i]:
                    dp[i] = dp[i - val] + 1
        ans = []
        i = n
        while i:
            for num in nums:
                val = score[num - 1]
                if i >= val and dp[i] == dp[i - val] + 1:
                    ans.append(num)
                    i -= val
                    break
        ans.sort(reverse=True)
        ac.st("".join(str(x) for x in ans))
        return

    @staticmethod
    def abc_145e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc145/tasks/abc145_e
        tag: brain_teaser|bag_dp|finite|sort|refresh_table
        """
        n, t = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort()
        dp = [0] * (t + 3010)
        for x, y in nums:
            for i in range(t - 1, -1, -1):
                if dp[i] + y > dp[i + x]:
                    dp[i + x] = dp[i] + y
        ac.st(max(dp))
        return

    @staticmethod
    def ac_6(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/6/
        tag: monotonic_queue|matrix_bag_dp|classical|classical
        """

        n, m = ac.read_list_ints()
        dp = [0] * (m + 1)
        for _ in range(n):
            # value weight number
            v, w, s = ac.read_list_ints()
            for r in range(v):
                stack = deque()
                for i in range(r, m + 1, v):
                    while stack and stack[0][0] < i - s * v:
                        stack.popleft()
                    while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                        stack.pop()
                    stack.append([i, dp[i]])
                    dp[i] = stack[0][1] + (i - stack[0][0]) // v * w
        ac.st(dp[-1])
        return

    @staticmethod
    def ac_10(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/10/
        tag: tree_bag_dp|classical
        """

        n, m = ac.read_list_ints()
        vol = []
        weight = []
        parent = [-1] * n
        dct = [[] for _ in range(n)]
        root = 0
        for i in range(n):
            v, w, p = ac.read_list_ints()
            p -= 1
            parent[i] = p
            if p != -2:
                dct[p].append(i)
            else:
                root = i
            vol.append(v)
            weight.append(w)

        stack = [root]
        sub = [[0] * (m + 1) for _ in range(n)]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                sub[i][vol[i]] = weight[i]
                for j in dct[i]:
                    cur = sub[i][:]
                    for x in range(vol[i], m + 1):
                        for y in range(m + 1 - x):
                            cur[x + y] = max(cur[x + y], sub[i][x] + sub[j][y])
                    sub[i] = cur[:]
        ac.st(max(sub[root]))
        return

    @staticmethod
    def ac_11(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/11/
        tag: bag_dp|counter|specific_plan
        """

        n, m = ac.read_list_ints()
        dp = [0] * (m + 1)
        cnt = [1] * (m + 1)
        mod = 10 ** 9 + 7
        for _ in range(n):
            v, w = ac.read_list_ints()
            for i in range(m, v - 1, -1):
                if dp[i - v] + w > dp[i]:
                    dp[i] = dp[i - v] + w
                    cnt[i] = cnt[i - v]
                elif dp[i - v] + w == dp[i]:
                    cnt[i] += cnt[i - v]
                    cnt[i] %= mod
        ac.st(cnt[-1])
        return

    @staticmethod
    def ac_12_1(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/12/
        tag: bag_dp|specific_plan|finite|lexicographical_order
        """

        n, m = ac.read_list_ints()
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        nums = [ac.read_list_ints() for _ in range(n)]

        for i in range(n - 1, -1, -1):
            v, w = nums[i]
            for j in range(m, -1, -1):
                dp[i][j] = dp[i + 1][j]
                if j >= v and dp[i + 1][j - v] + w > dp[i][j]:
                    dp[i][j] = dp[i + 1][j - v] + w

        j = m
        path = []
        for i in range(n):
            v, w = nums[i]
            if j >= v and dp[i][j] == dp[i + 1][j - v] + w:
                j -= v
                path.append(i + 1)
        ac.lst(path)
        return

    @staticmethod
    def ac_12_2(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/12/
        tag: bag_dp|specific_plan|finite
        """

        n, m = ac.read_list_ints()
        dp = [[0, [-1]] for _ in range(m + 1)]
        for ind in range(n):
            v, w = ac.read_list_ints()
            for i in range(m, v - 1, -1):
                if dp[i - v][0] + w > dp[i][0] or (
                        dp[i - v][0] + w == dp[i][0] and dp[i - v][1] + [ind + 1] < dp[i][1]):
                    dp[i] = [dp[i - v][0] + w, dp[i - v][1] + [ind + 1]]
        ac.lst(dp[-1][1][1:])
        return

    @staticmethod
    def lg_p1064(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1064
        tag: bag_dp|finite|brute_force|classification_discussion|group_bag_dp
        """

        n, m = ac.read_list_ints()
        dct = [[] for _ in range(m)]
        sub = [[] for _ in range(m)]
        for i in range(m):
            v, p, q = ac.read_list_ints()
            if q == 0:
                dct[i].append([v, p])
            else:
                sub[q - 1].append([v, p])
        dp = [[0] * (n + 1) for _ in range(2)]
        pre = 0
        for i in range(m):
            if dct[i]:
                cur = 1 - pre
                dp[cur] = dp[pre][:]
                x = len(sub[i])
                for j in range(1 << x):
                    lst = dct[i] + [sub[i][k] for k in range(x) if j & (1 << k)]
                    gain = sum(v * p for v, p in lst)
                    cost = sum(v for v, _ in lst)
                    for xx in range(n, cost - 1, -1):
                        dp[cur][xx] = ac.max(
                            dp[cur][xx], dp[pre][xx - cost] + gain)
                pre = cur
        ac.st(dp[pre][-1])
        return

    @staticmethod
    def lg_p1156(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1156
        tag: bag_dp|finite
        """

        n, m = ac.read_list_ints()

        dct = [ac.read_list_ints() for _ in range(m)]
        dct.sort(key=lambda it: it[0])

        dp = [-inf] * (n + 1)  # dp[height]=life
        dp[0] = 10
        for t, f, h in dct:
            if dp[0] < t:
                ac.st(dp[0])
                return
            for i in range(n, -1, -1):
                if dp[i] >= t:
                    if i + h >= n:
                        ac.st(t)
                        return
                    if i + h <= n:
                        dp[i + h] = ac.max(dp[i + h], dp[i])
                    dp[i] += f
        ac.st(dp[0])
        return

    @staticmethod
    def lg_p1273(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1273
        tag: tree|graph|group_bag_dp
        """

        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for j in range(n - m):
            lst = ac.read_list_ints()
            for i in range(1, len(lst), 2):
                dct[j].append([lst[i] - 1, lst[i + 1]])
        nums = [0] * (n - m) + ac.read_list_ints()
        sub = [[] for _ in range(n)]
        stack = [0]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j, _ in dct[i]:
                    stack.append(j)
            else:
                i = ~i
                sub[i].append(0)
                if i >= n - m:
                    sub[i].append(nums[i])
                    continue

                for j, cost in dct[i]:
                    cur = sub[i][:]
                    for k1 in range(m + 1):
                        if k1 >= len(sub[i]):
                            break
                        for k2 in range(m - k1 + 1):
                            if k2 >= len(sub[j]):
                                break
                            if len(cur) < k1 + k2 + 1:
                                cur.extend([-inf] * (k1 + k2 + 1 - len(cur)))
                            cur[k1 + k2] = ac.max(cur[k1 + k2], sub[j][k2] + sub[i][k1] - cost)
                    sub[j] = []
                    sub[i] = cur[:]
        for x in range(m, -1, -1):
            if x < len(sub[0]) and sub[0][x] >= 0:
                ac.st(x)
                return
        ac.st(0)
        return

    @staticmethod
    def lg_p1284(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1284
        tag: brute_force|triangle|math|bag_dp|classical|hard
        """

        n = ac.read_int()

        def check():
            ss = (a + b + c) / 2
            return (ss * (ss - a) * (ss - b) * (ss - c)) ** 0.5

        def idx(i1, j1):
            return i1 * (s // 2 + 1) + j1

        nums = []
        while len(nums) < n:
            nums.extend(ac.read_list_ints())

        s = sum(nums)
        dp = [0] * (s // 2 + 1) * (s // 2 + 1)
        dp[0] = 1
        for num in nums:
            for i in range(s // 2, -1, -1):
                for j in range(s // 2, -1, -1):
                    if j >= num and dp[idx(i, j - num)]:
                        dp[idx(i, j)] = 1
                    if i >= num and dp[idx(i - num, j)]:
                        dp[idx(i, j)] = 1
        ans = -1
        for a in range(s // 2 + 1):
            for b in range(s // 2 + 1):
                if dp[idx(a, b)]:
                    c = s - a - b
                    if b + c > a > 0 and a + c > b > 0 and a + b > c > 0:
                        cur = check()
                        ans = ac.max(ans, cur)
        if ans == -1:
            ac.st(ans)
        else:
            ac.st(int(ans * 100))
        return

    @staticmethod
    def lg_p1441(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1441
        tag: brute_force|bag_dp
        """
        n, m = ac.read_list_ints()
        a = ac.read_list_ints()
        ans = 0
        s = sum(a)
        for item in combinations(a, n - m):
            dp = [0] * (s + 1)
            dp[0] = 1
            for num in item:
                for i in range(s, num - 1, -1):
                    if dp[i - num]:
                        dp[i] = 1
            cur = sum(dp) - 1
            ans = ac.max(ans, cur)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1537(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1537
        tag: bin_split|bag_dp
        """
        case = 0
        while True:
            lst = ac.read_list_ints()
            if sum(lst) == 0:
                break

            case += 1
            ac.st(f"Collection #{case}:")
            s = sum(lst[i] * (i + 1) for i in range(6))
            if s % 2:
                ac.st("Can't be divided.")
                ac.st("")
                continue

            m = s // 2
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in range(6):
                w, s = x + 1, lst[x]
                if s:
                    for num in BagDP().bin_split_1(s):
                        for i in range(m, w * num - 1, -1):
                            if dp[i - num * w]:
                                dp[i] = 1
            if dp[-1]:
                ac.st("Can be divided.")
            else:
                ac.st("Can't be divided.")
            ac.st("")
        return

    @staticmethod
    def lg_p1541(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1541
        tag: brute_force|matrix_dp|fill_table
        """

        def idx(i1, i2, i3, i4):
            return i1 * (b + 1) * (c + 1) * (d + 1) + i2 * (c + 1) * (d + 1) + i3 * (d + 1) + i4

        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        cnt = Counter(ac.read_list_ints())
        a, b, c, d = cnt[1], cnt[2], cnt[3], cnt[4]
        dp = [0] * (a + 1) * (b + 1) * (c + 1) * (d + 1)
        dp[0] = nums[0]
        ans = 0
        for i in range(a + 1):
            for j in range(b + 1):
                for k in range(c + 1):
                    for p in range(d + 1):
                        if i + 2 * j + 3 * k + 4 * p <= n - 1:
                            pre = 0
                            if i:
                                pre = ac.max(pre, dp[idx(i - 1, j, k, p)])
                            if j:
                                pre = ac.max(pre, dp[idx(i, j - 1, k, p)])
                            if k:
                                pre = ac.max(pre, dp[idx(i, j, k - 1, p)])
                            if p:
                                pre = ac.max(pre, dp[idx(i, j, k, p - 1)])
                            dp[idx(i, j, k, p)] = ac.max(dp[idx(i, j, k, p)],
                                                                pre + nums[i + 2 * j + 3 * k + 4 * p])
                        if i + 2 * j + 3 * k + 4 * p == n - 1:
                            ans = ac.max(ans, dp[idx(i, j, k, p)])
        ac.st(ans)
        return

    @staticmethod
    def lg_p1759(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1759
        tag: matrix_bag_dp|lexicographical_order|specific_plan
        """
        m, v, n = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        dp = [[[0, []] for _ in range(v + 1)] for _ in range(m + 1)]
        for i in range(n):
            a, b, c = nums[i]
            for j in range(m, a - 1, -1):
                for k in range(v, b - 1, -1):
                    t, p = dp[j - a][k - b]
                    if dp[j][k][0] < t + c or (dp[j][k][0] == t + c and p + [i + 1] < dp[j][k][1]):
                        dp[j][k] = [t + c, p + [i + 1]]
        ans1, ans2 = dp[m][v]
        ac.st(ans1)
        ac.lst(ans2)
        return

    @staticmethod
    def lg_p1776(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1776
        tag: matrix_bag_dp|bin_split|monotonic_queue|classical
        """

        n, m = ac.read_list_ints()
        dp = [0] * (m + 1)
        for _ in range(n):
            a, b, c = ac.read_list_ints()
            v, w, s = b, a, c
            for r in range(v):
                stack = deque()
                for i in range(r, m + 1, v):
                    while stack and stack[0][0] < i - s * v:
                        stack.popleft()
                    while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                        stack.pop()
                    stack.append([i, dp[i]])
                    dp[i] = stack[0][1] + (i - stack[0][0]) // v * w
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p1799(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1799
        tag: matrix_bag_dp
        """

        n = ac.read_int()
        if not n:
            ac.st(0)
            return
        nums = ac.read_list_ints()
        dp = [[-inf] * (n + 1) for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = 1 if nums[0] == 1 else 0
        for i in range(1, n):
            dp[i][0] = 0
            for j in range(1, i + 2):
                dp[i][j] = ac.max(dp[i - 1][j], dp[i - 1][j - 1] + int(nums[i] == j))
        ac.st(max(dp[n - 1]))
        return

    @staticmethod
    def lg_p1833(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1833
        tag: infinite|bag_dp|monotonic_queue|matrix_bag_dp|classical
        """

        def check(st):
            hh, mm = st.split(":")
            return int(hh) * 60 + int(mm)

        s, e, n = ac.read_list_strs()
        t = check(e) - check(s)
        dp = [0] * (t + 1)
        for _ in range(int(n)):
            tt, cc, p = ac.read_list_ints()
            if not p:
                for i in range(tt, t + 1):
                    dp[i] = ac.max(dp[i], dp[i - tt] + cc)
            else:
                v, w, s = tt, cc, p
                for r in range(v):
                    stack = deque()
                    for i in range(r, t + 1, v):
                        while stack and stack[0][0] < i - s * v:
                            stack.popleft()
                        while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                            stack.pop()
                        stack.append([i, dp[i]])
                        dp[i] = stack[0][1] + (i - stack[0][0]) // v * w
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p2014(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2014
        tag: dag|tree_bag_dp
        """

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
                        for x in range(m + 2):
                            for y in range(m + 2 - x):
                                cur[x + y] = ac.max(cur[x + y], dp[i][x] + dp[j][y])
                        dp[i] = cur[:]
        ac.st(dp[0][m + 1])
        return

    @staticmethod
    def lg_p2079(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2079
        tag: rolling_hash|bag_dp|classical
        """

        n, v = ac.read_list_ints()
        dp = [defaultdict(lambda: defaultdict(lambda: -inf)), defaultdict(lambda: defaultdict(lambda: -inf))]
        pre = 0
        dp[pre][0][0] = 0
        for i in range(n):
            c, x, y = ac.read_list_ints()
            cur = 1 - pre
            for c1 in dp[pre]:
                for x1 in dp[pre][c1]:
                    if c1 + c <= v:
                        dp[cur][c1 + c][x1 + x] = ac.max(dp[cur][c1 + c][x1 + x], dp[pre][c1][x1] + y)
                    dp[cur][c1][x1] = ac.max(dp[cur][c1][x1], dp[pre][c1][x1])
            pre = cur
        ans = -inf
        for c1 in dp[pre]:
            for x1 in dp[pre][c1]:
                if x1 >= 0:
                    ans = ac.max(ans, dp[pre][c1][x1])
        ac.st(ans)
        return

    @staticmethod
    def lg_p2170(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2170
        tag: union_find|bag_dp|finite|bin_split
        """

        n, m, k = ac.read_list_ints()
        uf = UnionFind(n)
        for _ in range(k):
            i, j = ac.read_list_ints_minus_one()
            uf.union(i, j)
        dct = defaultdict(int)
        for i in range(n):
            dct[uf.find(i)] += 1
        lst = list(dct.values())
        del uf

        target = ac.min(2 * m, n)
        dp = [0] * (target + 1)
        dp[0] = 1
        cnt = Counter(lst)
        for num in cnt:
            for x in BagDP().bin_split_1(cnt[num]):
                for i in range(target, x * num - 1, -1):
                    if dp[i - x * num]:
                        dp[i] = 1
        ans = 0
        for i in range(1, target + 1):
            if dp[i] and abs(i - m) < abs(ans - m):
                ans = i
        ac.st(ans)
        return

    @staticmethod
    def lg_p2214(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2214
        tag: bag_dp|greedy
        """

        n, b = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(b)]
        voice = [ac.read_int() for _ in range(n)]

        for i in range(n - 1, 0, -1):
            if voice[i - 1] > 0:
                voice[i] -= voice[i - 1] - 1
        ceil = max(voice)
        if any(v < 0 for v in voice):
            ac.st(-1)
            return

        dp = [inf] * (ceil + 1)
        dp[0] = 0
        for num in nums:
            for i in range(num, ceil + 1):
                dp[i] = ac.min(dp[i - num] + 1, dp[i])
        ans = sum(dp[x] for x in voice)
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lg_p2306(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2306
        tag: data_range|counter|finite|bin_split
        """

        n, m, k = ac.read_list_ints()
        cnt = defaultdict(lambda: defaultdict(int))
        for _ in range(n):
            a, b = ac.read_list_ints()
            cnt[a][b] += 1
        dp = [0] * (m + 1)
        for a in cnt:
            for b in cnt[a]:
                for x in BagDP().bin_split_1(cnt[a][b]):
                    for i in range(m, x * a - 1, -1):
                        dp[i] = ac.max(dp[i], dp[i - x * a] + x * b)
        ans = max(dp)
        if ans >= k:
            ac.st("yes")
        else:
            ac.st("no")
        ac.st(ans)
        return

    @staticmethod
    def lg_p2320(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2320
        tag: bin_split|greedy|reverse_thinking
        """

        m = ac.read_int()
        ans = []
        while m:
            ans.append((m + 1) // 2)
            m //= 2
        ac.st(len(ans))
        ac.st(*ans[::-1])
        return

    @staticmethod
    def lg_p2737(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2737
        tag: infinite|bag_dp
        """

        n = ac.read_int()
        ceil = 256 ** 2 + 1
        nums = [ac.read_int() for _ in range(n)]
        dp = [0] * (ceil + 1)
        dp[0] = 1
        for i in range(1, ceil + 1):
            for num in nums:
                if i >= num and dp[i - num]:
                    dp[i] = 1
        ans = 0
        for i in range(1, ceil + 1):
            if not dp[i]:
                ans = i
        ac.st(ans if ans < ceil else 0)
        return

    @staticmethod
    def lg_p2760(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2760
        tag: monotonic_queue|matrix_bag_dp|classical
        """

        m, n, p, t = ac.read_list_ints()
        rest = ac.min(p, t - 1)
        dp = [0] * (rest + 1)
        grid = [ac.read_list_ints() for _ in range(m)]
        mat = [ac.read_list_ints() for _ in range(m)]
        for a in range(m):
            for b in range(n):
                if grid[a][b]:
                    v, w, s = (a + 1 + b + 1) * 2, grid[a][b], mat[a][b]
                    for r in range(v):
                        stack = deque()
                        for i in range(r, rest + 1, v):
                            while stack and stack[0][0] < i - s * v:
                                stack.popleft()
                            while stack and stack[-1][1] + (i - stack[-1][0]) // v * w <= dp[i]:
                                stack.pop()
                            stack.append([i, dp[i]])
                            dp[i] = stack[0][1] + (i - stack[0][0]) // v * w
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p2854(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2854
        tag: bag_dp|group_bag_dp|finite
        """

        length, n, b = ac.read_list_ints()
        dp = [[-inf] * (b + 1) for _ in range(length + 1)]
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort()
        for x, w, f, c in nums:
            if x == 0:
                if c <= b:
                    dp[x + w][c] = ac.max(dp[x + w][c], f)
            else:
                for i in range(b + 1):
                    if i + c <= b and x + w <= length:
                        dp[x + w][i + c] = ac.max(dp[x + w][i + c], dp[x][i] + f)
        ans = max(dp[length])
        ac.st(ans if ans > -inf else -1)
        return

    @staticmethod
    def lg_p2938(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2938
        tag: infinite|group_bag_dp
        """

        s, d, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(s)]
        for i in range(1, d):
            dp = [0] * (m + 1)
            for j in range(s):
                a, b = nums[j][i - 1], nums[j][i]
                if b > a:
                    for p in range(a, m + 1):
                        dp[p] = ac.max(dp[p], dp[p - a] + b)
            m = max(m - i + dp[i] for i in range(m + 1))
        ac.st(m)
        return

    @staticmethod
    def lg_p2979(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2979
        tag: bag_dp|group_bag_dp|finite
        """

        n, t, k = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        m = 5 * t // 4 + 1

        dp1 = [0] * (m + 1)
        for v, h in nums:
            for i in range(h, m + 1):
                if dp1[i - h] + v > dp1[i]:
                    dp1[i] = dp1[i - h] + v
        ans = dp1[t]
        for v, h in nums:
            if h >= k:
                for i in range(t, h - 1, -1):
                    ans = ac.max(ans, dp1[(i - h) * 5 // 4] + v)
        ac.st(ans)
        return

    @staticmethod
    def lg_p3010(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3010
        tag: bag_dp|heapq|specific_plan
        """

        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        s = sum(nums)
        mod = 10 ** 6

        t = s // 2
        dp = [0] * (t + 1)
        dp[0] = 1
        cnt = [0] * (t + 1)
        cnt[0] = 1
        for num in nums:
            for i in range(t, num - 1, -1):
                if dp[i - num]:
                    dp[i] = 1
                    cnt[i] += cnt[i - num]
                    cnt[i] %= mod

        for i in range(t, -1, -1):
            if dp[i]:
                ac.st(s - 2 * i)
                ac.st(cnt[i])
                break
        return

    @staticmethod
    def lg_p3423(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3423
        tag: bin_split|matrix_bag_dp|specific_plan
        """

        n = ac.read_int()
        b = ac.read_list_ints()
        c = ac.read_list_ints()
        k = ac.read_int()
        dp = [inf] * (k + 1)
        dp[0] = 0
        state = [[] for _ in range(k + 1)]
        for j in range(n):
            bb, cc = b[j], c[j]
            for x in BagDP().bin_split_1(cc):
                for i in range(k, x * bb - 1, -1):
                    if dp[i - x * bb] + x < dp[i]:
                        dp[i] = dp[i - x * bb] + x
                        state[i] = state[i - x * bb][:] + [[bb, x]]
        cnt = defaultdict(int)
        for bb, xx in state[k]:
            cnt[bb] += xx
        ac.st(dp[k])
        ac.lst([cnt[bb] for bb in b])
        return

    @staticmethod
    def lg_p3983(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3983
        tag: infinite|bag_dp
        """

        n = ac.read_int()
        m = 10
        a = [0] + ac.read_list_ints()
        for i in range(1, m + 1):
            for j in range(i + 1):
                a[i] = ac.max(a[i], a[j] + a[i - j])

        cost = [0] + [1, 3, 5, 7, 9, 10, 11, 14, 15, 17]
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(i, n + 1):
                dp[j] = ac.max(dp[j], dp[j - i] + a[i] - cost[i])
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p5322(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5322
        tag: matrix_dp|group_bag_dp|classical
        """

        s, n, m = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(s)]
        dp = [0] * (m + 1)
        for i in range(n):
            lst = [grid[x][i] for x in range(s)]
            lst.sort()
            for j in range(m, -1, -1):
                for ind, w in enumerate(lst):
                    if j <= w * 2:
                        break
                    dp[j] = ac.max(dp[j], dp[j - 2 * w - 1] + (ind + 1) * (i + 1))
        ac.st(dp[-1])
        return

    @staticmethod
    def lg_p5365(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5365
        tag: bag_dp|infinite|brute_force|counter
        """
        n, m = ac.read_list_ints()
        kk = ac.read_list_ints()
        cc = ac.read_list_ints()
        s = sum(kk[i] * cc[i] for i in range(n))
        dp = [0] * (s + 1)
        dp[0] = 1
        for i in range(n):
            k, c = kk[i], cc[i]
            for x in range(s, -1, -1):
                for p in range(1, k + 1):
                    if x < p * c:
                        break
                    dp[x] = ac.max(dp[x], dp[x - p * c] * p)
        for i in range(s + 1):
            if dp[i] >= m:
                ac.st(i)
                break
        return

    @staticmethod
    def lg_p5662(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5662
        tag: infinite|bag_dp|greedy
        """

        t, n, m = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(t)]
        for i in range(1, t):
            dp = [0] * (m + 1)
            for j in range(n):
                b, a = grid[i][j], grid[i - 1][j]
                if b > a:
                    for x in range(a, m + 1):
                        dp[x] = ac.max(dp[x], dp[x - a] + b)
            m = max(m - i + dp[i] for i in range(m + 1))
        ac.st(m)
        return

    @staticmethod
    def lg_p1417(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1417
        tag: greedy|sort|bag_dp
        """
        t, n = ac.read_list_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        c = ac.read_list_ints()
        dp = [0] * (t + 1)
        ind = list(range(n))
        ind.sort(key=lambda it: -b[it] / c[it])
        for i in ind:
            aa, bb, cc = a[i], b[i], c[i]
            for j in range(t, cc - 1, -1):
                dp[j] = ac.max(dp[j], dp[j - cc] + aa - j * bb)
        ac.st(max(dp))
        return

    @staticmethod
    def ac_4081(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4084/
        tag: matrix_bag_dp
        """

        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()

        def check2(xx):
            res = 0
            while xx % 2 == 0:
                res += 1
                xx //= 2
            return res

        def check5(xx):
            res = 0
            while xx % 5 == 0:
                res += 1
                xx //= 5
            return res

        cnt2 = [check2(num) for num in nums]
        cnt5 = [check5(num) for num in nums]

        s5 = sum(cnt5)
        dp = [[-inf] * (s5 + 1) for _ in range(k + 1)]
        dp[0][0] = 0
        for i in range(n):
            a2 = cnt2[i]
            a5 = cnt5[i]
            for j in range(k, 0, -1):
                for p in range(s5, a5 - 1, -1):
                    x, y = dp[j][p], dp[j - 1][p - a5] + a2
                    if y > x:
                        dp[j][p] = y
        ans = 0
        for a5 in range(s5 + 1):
            cur = ac.min(dp[k][a5], a5)
            if cur > ans:
                ans = cur
        ac.st(ans)
        return

    @staticmethod
    def lc_2902(nums: List[int], ll: int, r: int) -> int:
        """
        url: https://leetcode.cn/problems/count-of-sub-multisets-with-bounded-sum/
        tag: monotonic_queue|mod|group_bag_dp|prefix_sum|inclusion_exclusion|lazy_bag_like
        """
        cnt = Counter(nums)
        mod = 10 ** 9 + 7
        dp = [0] * (r + 1)
        dp[0] = 1
        for num in cnt:
            if num:
                c = cnt[num]
                for i in range(num):
                    pre = [0]
                    x = 0
                    for j in range(i, r + 1, num):
                        val = pre[-1] + dp[j]
                        dp[j] += pre[x]
                        if x - c >= 0:
                            dp[j] -= pre[x - c]
                        dp[j] %= mod
                        pre.append(val % mod)
                        x += 1
        return sum(dp[ll:]) * (cnt[0] + 1) % mod

    @staticmethod
    def lc_1049(stones: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/last-stone-weight-ii/
        tag: bag_dp
        """
        s = sum(stones)
        dp = [0] * (s // 2 + 1)
        dp[0] = 1
        for num in stones:
            for i in range(s // 2, num - 1, -1):
                if dp[i - num]:
                    dp[i] = 1
        return min(abs(s - 2 * i) for i in range(s // 2 + 1) if dp[i])
