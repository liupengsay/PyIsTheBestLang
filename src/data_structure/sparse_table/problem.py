"""
Algorithm：sparse_table|multiplication_method|aggregation_property|sub_consequence|log_trick
Description：static_range|range_query|range_max|range_min|range_gcd|range_and|range_or|range_lcm|sparse_table|binary_search

====================================LeetCode====================================
1521（https://leetcode.cn/problems/find-a-value-of-a-mysterious-function-closest-to-target/）bit_operation|sub_consequence
2411（https://leetcode.cn/problems/smallest-subarrays-with-maximum-bitwise-or/）sub_consequence|range_or
2447（https://leetcode.cn/problems/number-of-subarrays-with-gcd-equal-to-k/）range_gcd|counter|sub_consequence
2470（https://leetcode.cn/problems/number-of-subarrays-with-lcm-equal-to-k/）range_lcm|counter|sub_consequence
2654（https://leetcode.cn/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/）range_gcd|sub_consequence
2836（https://leetcode.cn/problems/maximize-value-of-function-in-a-ball-passing-game/description/）multiplication_method|classical
3117（https://leetcode.com/problems/minimum-sum-of-values-by-dividing-array/）range_and|linear_dp|classical
100315（https://leetcode.cn/problems/find-subarray-with-bitwise-and-closest-to-k/）range_and|bit_operation|classical|sparse_table|binary_search|binary_right
2398（https://leetcode.cn/problems/maximum-number-of-robots-within-budget/）sparse_table|binary_search|monotonic_queue

=====================================LuoGu======================================
P3865（https://www.luogu.com.cn/problem/P3865）sparse_table|range_max
P2880（https://www.luogu.com.cn/problem/P2880）sparse_table|range_max|range_min
P3865（https://www.luogu.com.cn/problem/P3865）sparse_table|range_gcd
P1816（https://www.luogu.com.cn/problem/P1816）sparse_table|range_min
P2412（https://www.luogu.com.cn/problem/P2412）lexicographical_order|sparse_table
P5097（https://www.luogu.com.cn/problem/P5097）sparse_table|range_min
P5648（https://www.luogu.com.cn/problem/P5648）sparse_table|range_max_index|monotonic_stack
P2048（https://www.luogu.com.cn/problem/P2048）sparse_table_index|heapq|greedy

===================================CodeForces===================================
1691D（https://codeforces.com/problemset/problem/1691/D）monotonic_stack|brute_force|sparse_table|range_max|range_min
689D（https://codeforces.com/problemset/problem/689/D）binary_search|sparse_table|counter
1359D（https://codeforces.com/problemset/problem/1359/D）monotonic_stack|brute_force|sparse_table|range_max|range_min
1548B（https://codeforces.com/problemset/problem/1548/B）sparse_table|range_gcd|brute_force|binary_search
474F（https://codeforces.com/problemset/problem/474/F）sparse_table|range_min|range_gcd|binary_search|counter
1834E（https://codeforces.com/contest/1834/problem/E）sparse_table|range_lcm
1878E（https://codeforces.com/contest/1878/problem/E）sparse_table|range_and
1547F（https://codeforces.com/contest/1547/problem/F）sparse_table|range_gcd
1579F（https://codeforces.com/contest/1579/problem/F）circular_section|range_and
1709D（https://codeforces.com/contest/1709/problem/D）sparse_table|range_max|implemention
1516D（https://codeforces.com/contest/1516/problem/D）multiplication_method
1977C（https://codeforces.com/contest/1977/problem/C）data_range|subsequence_lcm|brain_teaser|classical
1847F（https://codeforces.com/contest/1847/problem/F）range_or|classical|implemention|sub_consequence
359D（https://codeforces.com/problemset/problem/359/D）range_gcd|classical
475D（https://codeforces.com/problemset/problem/475/D）range_gcd|classical
451D（https://codeforces.com/problemset/problem/451/D）sparse_table|divide_and_conquer
1632D（https://codeforces.com/problemset/problem/1632/D）sparse_table|binary_search

=====================================AcWing=====================================
109（https://www.acwing.com/problem/content/111/）greedy|multiplication_method

=====================================AtCoder=====================================
ABC212F（https://atcoder.jp/contests/abc212/tasks/abc212_f）multiplication_method|build_graph|brain_teaser|classical
ABC367E（https://atcoder.jp/contests/abc367/tasks/abc367_e）multiplication_method|classical

"""

import bisect
import math
from collections import defaultdict, Counter
from functools import lru_cache
from heapq import heappop, heapify, heappush
from itertools import accumulate
from operator import and_
from typing import List

from src.data_structure.sparse_table.template import SparseTable, SparseTableIndex
from src.mathmatics.prime_factor.template import PrimeFactor
from src.utils.fast_io import FastIO



class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p2880(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2880
        tag: sparse_table|range_max|range_min
        """
        n, q = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]
        st1 = SparseTable(nums, max)
        st2 = SparseTable(nums, min)
        for _ in range(q):
            a, b = ac.read_list_ints_minus_one()
            ac.st(st1.query(a, b) - st2.query(a, b))
        return

    @staticmethod
    def lg_p3865(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3865
        tag: sparse_table|range_gcd
        """
        n, m = ac.read_list_ints()
        st = SparseTable(ac.read_list_ints(), max)
        for _ in range(m):
            x, y = ac.read_list_ints_minus_one()
            ac.st(st.query(x, y))
        return

    @staticmethod
    def cf_474f(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/474/F
        tag: sparse_table|range_min|range_gcd|binary_search|counter
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        dct = defaultdict(list)
        for i, num in enumerate(nums):
            dct[num].append(i)
        st_gcd = SparseTable(nums, math.gcd)
        st_min = SparseTable(nums, ac.min)
        for _ in range(ac.read_int()):
            x, y = ac.read_list_ints_minus_one()
            num1 = st_gcd.query(x, y)
            num2 = st_min.query(x, y)
            if num1 == num2:
                res = bisect.bisect_right(dct[num1], y) - bisect.bisect_left(dct[num1], x)
                ac.st(y - x + 1 - res)
            else:
                ac.st(y - x + 1)
        return

    @staticmethod
    def ac_109(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/111/
        tag: greedy|multiplication_method|two_pointers|merge_sort|hard|classical
        """

        def range_merge_to_disjoint(lst1, lst2):
            a, b = len(lst1), len(lst2)
            x = y = 0
            res = []
            while x < a or y < b:
                if x == a or (y < b and lst2[y] < lst1[x]):
                    res.append(lst2[y])
                    y += 1
                else:
                    res.append(lst1[x])
                    x += 1
            return res

        def check(lst1):
            k = len(lst1)
            x, y = 0, k - 1
            res = cnt = 0
            while x < y and cnt < m:
                res += (lst1[x] - lst1[y]) ** 2
                if res > t:
                    return False
                x += 1
                y -= 1
                cnt += 1
            return True

        for _ in range(ac.read_int()):
            n, m, t = ac.read_list_ints()
            nums = ac.read_list_ints()
            ans = i = 0
            while i < n:
                p = 1
                lst = [nums[i]]
                right = i
                while p and right < n:
                    cur = nums[right + 1:right + p + 1]
                    cur.sort()
                    tmp = range_merge_to_disjoint(lst, cur)
                    if check(tmp):
                        lst = tmp[:]
                        right += p
                        p *= 2
                    else:
                        p //= 2
                ans += 1
                i = right + 1
            ac.st(ans)
        return

    @staticmethod
    def lg_p5648(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5648
        tag: sparse_table|range_max_index|monotonic_stack
        """
        n, t = ac.read_list_ints()
        nums = ac.read_list_ints()
        post = [n] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                post[stack.pop()] = i
            stack.append(i)
        edge = [[] for _ in range(n + 1)]
        for i in range(n):
            edge[post[i]].append(i)
        sub = [0] * (n + 1)
        stack = [n]
        while stack:
            i = stack.pop()
            for j in edge[i]:
                sub[j] = sub[i] + nums[j] * (i - j)
                stack.append(j)
        st = SparseTableIndex(nums, max)
        last_ans = 0
        for _ in range(t):
            u, v = ac.read_list_ints()
            left = 1 + (u ^ last_ans) % n
            q = 1 + (v ^ (last_ans + 1)) % (n - left + 1)
            right = left + q - 1
            ceil_ind = st.query(left - 1, right - 1)
            last_ans = sub[left - 1] - sub[ceil_ind] + nums[ceil_ind] * (right - ceil_ind)
            ac.st(last_ans)
        return

    @staticmethod
    def lc_2447(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-subarrays-with-gcd-equal-to-k/
        tag: range_gcd|counter|sub_consequence
        """
        ans = 0
        pre = dict()
        for num in nums:
            cur = dict()
            for p in pre:
                x = math.gcd(p, num)
                if x % k == 0:
                    cur[x] = cur.get(x, 0) + pre[p]
            if num % k == 0:
                cur[num] = cur.get(num, 0) + 1
            ans += cur.get(k, 0)
            pre = cur
        return ans

    @staticmethod
    def lc_2470(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/number-of-subarrays-with-lcm-equal-to-k/
        tag: range_lcm|counter|sub_consequence
        """
        ans = 0
        pre = dict()
        for num in nums:
            cur = dict()
            for p in pre:
                x = math.lcm(p, num)
                if k % x == 0:
                    cur[x] = cur.get(x, 0) + pre[p]
            if k % num == 0:
                cur[num] = cur.get(num, 0) + 1
            ans += cur.get(k, 0)
            pre = cur
        return ans

    @staticmethod
    def lc_2836(nex: List[int], k: int) -> int:
        """
        url:https://leetcode.cn/problems/maximize-value-of-function-in-a-ball-passing-game/
        tag: multiplication_method|classical|can_not_be_circular_section|can_not_to_be_permutation_circle
        """
        n = len(nex)
        ans = list(range(n))
        pos = list(range(n))
        s = nex[:]
        while k:
            if k & 1:
                ans = [ans[i] + s[pos[i]] for i in range(n)]
                pos = [nex[i] for i in pos]
            k >>= 1
            s = [s[i] + s[nex[i]] for i in range(n)]
            nex = [nex[i] for i in nex]
        return max(ans)

    @staticmethod
    def lc_2411(nums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/smallest-subarrays-with-maximum-bitwise-or/
        tag: sub_consequence|range_or
        """
        n = len(nums)
        ans = [0] * n
        post = dict()
        for i in range(n - 1, -1, -1):
            cur = dict()
            num = nums[i]
            for x in post:
                y = cur.get(x | num, inf)
                cur[x | num] = y if y < post[x] else post[x]
            cur[num] = i
            post = cur
            ans[i] = post[max(post)] - i + 1
        return ans

    @staticmethod
    def cf_1516d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1516/problem/D
        tag: multiplication_method|classical|sparse_table
        """
        pf = PrimeFactor(10 ** 5)
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        post = [n] * (n + 1)
        ind = [n] * (10 ** 5)
        for i in range(n - 1, -1, -1):
            right = post[i + 1]
            for p, _ in pf.prime_factor[nums[i]]:
                right = ac.min(ind[p], right)
                ind[p] = i
            post[i] = right

        col = max(2, math.ceil(math.log2(n)))
        dp = [[n] * col for _ in range(n)]
        for i in range(n):
            dp[i][0] = post[i]
        for j in range(1, col):
            for i in range(n):
                father = dp[i][j - 1]
                if father <= n - 1:
                    dp[i][j] = dp[father][j - 1]

        for _ in range(q):
            x, y = ac.read_list_ints_minus_one()
            ans = 0
            for j in range(col - 1, -1, -1):
                if dp[x][j] <= y:
                    x = dp[x][j]
                    ans += 1 << j
            ac.st(ans + 1)
        return

    @staticmethod
    def cf_1709d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1709/problem/D
        tag: sparse_table|range_max|implemention
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        st = SparseTable(nums, max)
        for _ in range(ac.read_int()):
            x1, y1, x2, y2, k = ac.read_list_ints()
            if x1 % k != x2 % k or y1 % k != y2 % k:
                ac.no()
                continue
            if y1 == y2:
                ac.yes()
                continue
            if y1 > y2:
                y1, y2 = y2, y1
            ceil = st.query(y1 - 1, y2 - 1)
            y = (n - x1) // k
            w = k * y + x1
            if w <= ceil:
                ac.no()
            else:
                ac.yes()
        return

    @staticmethod
    def cf_1878e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1878/problem/E
        tag: sparse_table|range_and
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            q = ac.read_int()
            query = [dict() for _ in range(n)]
            res = []
            for _ in range(q):
                ll, k = ac.read_list_ints()
                ll -= 1
                res.append([ll, k])
                query[ll][k] = -2
            post = dict()
            for i in range(n - 1, -1, -1):
                cur = dict()
                num = nums[i]
                for p in post:
                    x = p & num
                    if x not in cur or post[p] > cur[x]:
                        cur[x] = post[p]
                if num not in cur:
                    cur[num] = i
                lst = sorted(query[i].keys(), reverse=True)
                val = [[num, cur[num]] for num in cur]
                val.sort(reverse=True)
                right = -2
                m = len(val)
                p = 0
                for ke in lst:
                    while p < m and val[p][0] >= ke:
                        _, xx = val[p]
                        if xx > right:
                            right = xx
                        p += 1
                    query[i][ke] = right
                post = cur.copy()
            ac.lst([query[ll][k] + 1 for ll, k in res])
        return

    @staticmethod
    def lc_1521(arr: List[int], target: int) -> int:
        """
        url: https://leetcode.cn/problems/find-a-value-of-a-mysterious-function-closest-to-target/
        tag: bit_operation|sub_consequence
        """
        ans = abs(arr[0] - target)
        pre = {arr[0]}
        for num in arr[1:]:
            pre = {num & p for p in pre}
            pre.add(num)
            for x in pre:
                if abs(x - target) < ans:
                    ans = abs(x - target)
        return ans

    @staticmethod
    def lg_p2048(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2048
        tag: sparse_table_index|heapq|greedy
        """
        n, k, l, r = ac.read_list_ints()
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + ac.read_int()
        st_ind = SparseTableIndex(pre, max)
        stack = []
        for i in range(n):
            if i + l - 1 < n:
                j = st_ind.query(i + l, ac.min(i + r, n))
                stack.append((pre[i] - pre[j], i, j, i + l, ac.min(i + r, n)))
            else:
                break

        heapify(stack)
        ans = 0
        for _ in range(k):
            x, i, j, ll, rr = heappop(stack)
            ans -= x
            if ll <= j - 1:
                jj = st_ind.query(ll, j - 1)
                heappush(stack, (pre[i] - pre[jj], i, jj, ll, j - 1))
            if j + 1 <= rr:
                jj = st_ind.query(j + 1, rr)
                heappush(stack, (pre[i] - pre[jj], i, jj, j + 1, rr))
        ac.st(ans)
        return

    @staticmethod
    def cf_1548b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1548/B
        tag: sparse_table|range_gcd|two_pointers|diff_array|brain_teaser|classical
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            diff = [nums[i + 1] - nums[i] for i in range(n - 1)]
            st = SparseTable(diff, math.gcd)
            ans = 1
            j = 0
            for i in range(n - 1):
                if abs(diff[i]) == 1:
                    continue
                while j < i:
                    j += 1
                while j + 1 < n - 1 and st.query(i, j + 1) > 1:
                    j += 1
                if j - i + 2 > ans:
                    ans = j - i + 2
            ac.st(ans)
        return

    @staticmethod
    def cf_689d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1548/B
        tag: sparse_table|range_gcd|two_pointers|diff_array|brain_teaser|classical
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            diff = [nums[i + 1] - nums[i] for i in range(n - 1)]
            st = SparseTable(diff, math.gcd)
            ans = 1
            j = 0
            for i in range(n - 1):
                if abs(diff[i]) == 1:
                    continue
                while j < i:
                    j += 1
                while j + 1 < n - 1 and st.query(i, j + 1) > 1:
                    j += 1
                if j - i + 2 > ans:
                    ans = j - i + 2
            ac.st(ans)
        return

    @staticmethod
    def lc_3117(nums: List[int], and_values: List[int]) -> int:
        """
        url: https://leetcode.com/contest/weekly-contest-393/problems/minimum-sum-of-values-by-dividing-array/
        tag: range_and|linear_dp|classical
        """
        n = len(nums)
        m = len(and_values)

        @lru_cache(None)
        def dfs(i, pre, a):
            if i == n:
                return nums[-1] if pre == and_values[-1] and a == m else inf
            if a > m:
                return inf
            if pre == -1:
                res = dfs(i + 1, nums[i], a + 1)
                return res
            res = inf
            if pre == and_values[a - 1]:
                res = dfs(i + 1, nums[i], a + 1) + nums[i - 1]
            res = min(res, dfs(i + 1, nums[i] & pre, a))
            return res

        ans = dfs(0, -1, 0)
        return ans if ans < inf else -1

    @staticmethod
    def abc_212f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc212/tasks/abc212_f
        tag: multiplication_method|build_graph|brain_teaser|classical
        """

        def tuple_to_num(aa, bb, cc):
            return (aa << 34) | (bb << 2) | cc

        mask = (1 << 34) - 1

        def num_to_tp(num):
            return num >> 34, (num & mask) >> 2, num & 1

        n, m, q = ac.read_list_ints()
        pos = [[] for _ in range(n)]
        post = dict()
        for i in range(m):
            a, b, s, t = ac.read_list_ints()
            s = 2 * s + 1
            a -= 1
            b -= 1
            t = 2 * t + 1
            ss = tuple_to_num(a, s, 0)
            if ss not in post:
                post[ss] = dict()
            post[ss][0] = tuple_to_num(b, t, 1)
            pos[a].append((s, 0))
            pos[b].append((t, 1))

        lst = []
        for i in range(n):
            pos[i].sort(key=lambda it: (it[0], -it[1]))
            m = len(pos[i])
            ls = []
            for j in range(m - 1, -1, -1):
                if pos[i][j][1] == 0:
                    ls = pos[i][j][:]
                else:
                    if ls:
                        cur = tuple_to_num(i, pos[i][j][0], pos[i][j][1])
                        if cur not in post:
                            post[cur] = dict()
                        post[cur][0] = tuple_to_num(i, ls[0], ls[1])
            lst.append([ls[0] for ls in pos[i]])

        for x in range(1, 20):
            for s in post:
                if x - 1 in post[s] and post[s][x - 1] in post and x - 1 in post[post[s][x - 1]]:
                    post[s][x] = post[post[s][x - 1]][x - 1]

        for _ in range(q):
            x, y, z = ac.read_list_ints()
            z *= 2
            x *= 2
            y -= 1
            i = bisect.bisect_left(lst[y], x)
            if i == len(lst[y]):
                ac.st(y + 1)
                continue
            state = tuple_to_num(y, pos[y][i][0], pos[y][i][1])
            for x in range(19, -1, -1):
                if state in post and x in post[state] and num_to_tp(post[state][x])[1] <= z:
                    state = post[state][x]
            tmp = num_to_tp(state)
            if tmp[-1] == 0 and z > tmp[1]:
                res = num_to_tp(post[state][0])
                ac.lst([tmp[0] + 1, res[0] + 1])
            else:
                ac.st(tmp[0] + 1)
        return

    @staticmethod
    def cf_1977c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1977/problem/C
        tag: data_range|subsequence_lcm|brain_teaser|classical
        """
        ceil = 10 ** 9 + 1
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            ans = pre = 0
            dct = set(nums)
            cnt = Counter(nums)
            for num in nums:
                pre = math.lcm(pre, num) if pre else num
                if pre > ceil:
                    break
            if pre > ceil or pre not in dct:
                ans = n
                ac.st(ans)
                continue
            pre = defaultdict(int)
            pre[1] = 0
            for num in cnt:
                cur = pre.copy()
                for p in pre:
                    pp = math.lcm(p, num)
                    cur[pp] = max(cur[pp], pre[p] + cnt[num])
                pre = cur
                for k in pre:
                    if k not in dct:
                        ans = max(ans, pre[k])
            ac.st(ans)
        return

    @staticmethod
    def lc_100315(nums: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/find-subarray-with-bitwise-and-closest-to-k/
        tag: range_and|bit_operation|classical|sparse_table|binary_search|binary_right
        """
        n = len(nums)
        st = SparseTable(nums, and_)
        ans = inf
        initial = (1 << 32) - 1
        for i in range(n):
            j, val = st.bisect_right(i, k, initial)
            ans = min(ans, abs(val - k))
            if j + 1 < n:
                ans = min(ans, abs((val & nums[j + 1]) - k))
        return ans

    @staticmethod
    def cf_1847f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1847/problem/F
        tag: range_or|classical|implemention|sub_consequence
        """

        for _ in range(ac.read_int()):
            n, q = ac.read_list_ints()
            nums = ac.read_list_ints()

            def check(length, index):
                if length == 1:
                    return index + 1
                res = n + (length - 2) * (n - 1)
                start = (-length + 2) % n
                if index >= start:
                    res += index - start + 1
                else:
                    res += index + 1 + n - start
                return res

            val = dict()
            pre = dict()
            for i in range(2 * n):
                cur = dict()
                num = nums[i % n]
                for p in pre:
                    cur[p | num] = min(cur.get(p | num, inf), pre[p] + 1)
                cur[num] = min(cur.get(num, inf), 1)
                pre = cur
                for p in pre:
                    tmp = val.get(p, inf)
                    ind = check(pre[p], (i - pre[p] + 1) % n)
                    if ind < tmp:
                        val[p] = ind
            lst = [i << 32 | p for p, i in val.items()]
            lst.sort()
            ind = []
            val = []
            for pi in lst:
                i = pi >> 32
                p = pi ^ (i << 32)
                if val and val[-1] >= p:
                    continue
                val.append(p)
                ind.append(i)
            for _ in range(q):
                v = ac.read_int()
                if v >= val[-1]:
                    ac.st(-1)
                else:
                    i = bisect.bisect_right(val, v)
                    ac.st(ind[i])
        return

    @staticmethod
    def cf_359d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/359/D
        tag: range_gcd|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = [0] * n
        dct = defaultdict(int)
        for i, num in enumerate(nums):
            cur = defaultdict(int)
            for p in dct:
                x = math.gcd(p, num)
                cur[x] = max(cur[x], dct[p] + 1)
            cur[num] = max(cur[num], 1)
            pre[i] = cur[num]
            dct = cur

        post = [0] * n
        dct = defaultdict(int)
        for i in range(n - 1, -1, -1):
            num = nums[i]
            cur = defaultdict(int)
            for p in dct:
                x = math.gcd(p, num)
                cur[x] = max(cur[x], dct[p] + 1)
            cur[num] = max(cur[num], 1)
            post[i] = cur[num]
            dct = cur

        ceil = max(pre[i] + post[i] - 1 for i in range(n))
        ans = set()
        for i in range(n):
            if pre[i] + post[i] - 1 == ceil:
                ans.add(i - pre[i] + 2)
        ac.lst([len(ans), ceil - 1])

        ac.lst(sorted(ans))
        return

    @staticmethod
    def abc_367e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc367/tasks/abc367_e
        tag: multiplication_method|classical
        """
        n, k = ac.read_list_ints()
        x = ac.read_list_ints_minus_one()
        a = ac.read_list_ints()
        for xx in range(60):
            if k & (1 << xx):
                a = [a[x[i]] for i in range(n)]
            x = [x[x[i]] for i in range(n)]
        ac.lst(a)
        return

    @staticmethod
    def cf_451d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/451/D
        tag: sparse_table|divide_and_conquer
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            mx = SparseTableIndex(nums, max)
            mn = SparseTableIndex(nums, min)

            dct = dict()
            stack = [n - 1]
            while stack:
                val = stack.pop()
                if val >= 0:

                    ll, rr = val // n, val % n
                    if ll == rr:
                        dct[val] = 0
                        continue
                    stack.append(~val)
                    p1 = mx.query(ll, rr)
                    p2 = mn.query(ll, rr)
                    if p1 > p2:
                        p1, p2 = p2, p1
                    stack.append(ll * n + p1)
                    stack.append(p2 * n + rr)
                else:
                    val = ~val
                    ll, rr = val // n, val % n
                    p1 = mx.query(ll, rr)
                    p2 = mn.query(ll, rr)
                    if p1 > p2:
                        p1, p2 = p2, p1
                    dct[val] = dct[ll * n + p1] + dct[p2 * n + rr] + 1
            ans = dct[n - 1]
            ac.st(ans)
        return

    @staticmethod
    def cf_1632d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1632/D
        tag: sparse_table|binary_search
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        nums.reverse()
        st = SparseTable(nums, math.gcd)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            flag, j = st.bisect_right_length(i)
            if flag:
                dp[i] = dp[j + 1] + 1
            else:
                dp[i] = dp[i + 1]
        ac.lst(dp[:-1][::-1])
        return

    @staticmethod
    def lc_2398(charge: List[int], costs: List[int], budget: int) -> int:
        """
        url: https://leetcode.cn/problems/maximum-number-of-robots-within-budget/
        tag: sparse_table|binary_search|monotonic_queue
        """
        n = len(charge)
        m = n.bit_length()
        st = [[0] * n for _ in range(m + 1)]
        st[0] = charge[:]
        acc = list(accumulate(costs, initial=0))
        for i in range(1, m + 1):
            for j in range(n - (1 << i) + 1):
                st[i][j] = max(st[i - 1][j], st[i - 1][j + (1 << (i - 1))])
        ans = 0

        for i in range(n):
            pos = i
            pre = 0
            for x in range(m, -1, -1):
                if pos + (1 << x) - 1 < n and max(st[x][pos], pre) + (pos + (1 << x) - i) * (
                        acc[pos + (1 << x)] - acc[i]) <= budget:
                    pre = max(st[x][pos], pre)
                    pos += (1 << x)
            ans = max(ans, pos - i)
        return ans
