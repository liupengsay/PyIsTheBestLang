"""

Algorithm：binary_search
Description：monotonicity is necessary for solution like these, which always work together with SortedList, or can also use Bisect, sometimes with high precision
====================================LeetCode====================================
4（https://leetcode.com/problems/median-of-two-sorted-arrays/）binary_search|median|two_arrays|same_direction_pointer
81（https://leetcode.com/problems/search-in-rotated-sorted-array-ii/）binary_search|rotated_array|sorting
154（https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/）binary_search|rotated_array|sorting|duplicate_nums
162（https://leetcode.com/problems/find-peak-element/）binary_search|peak_index
2179（https://leetcode.com/problems/count-good-triplets-in-an-array/）binary_search|sorted_list
2141（https://leetcode.com/problems/maximum-running-time-of-n-computers/）greedy|binary_search|implemention
2102（https://leetcode.com/problems/sequentially-ordinal-rank-tracker/）binary_search|sorted_list
2563（https://leetcode.com/problems/count-the-number-of-fair-pairs/）binary_search|sorted_list
2604（https://leetcode.com/problems/minimum-time-to-eat-all-grains/）binary_search|greedy|pointer
1201（https://leetcode.com/problems/ugly-number-iii/）binary_search|counter|inclusion_exclusion
1739（https://leetcode.com/problems/building-boxes/）math|binary_search
1889（https://leetcode.com/problems/minimum-space-wasted-from-packaging/）sorting|prefix_sum|greedy|binary_search
2071（https://leetcode.com/problems/maximum-number-of-tasks-you-can-assign/）binary_search|greedy
2594（https://leetcode.com/problems/minimum-time-to-repair-cars/）binary_search
2517（https://leetcode.com/problems/maximum-tastiness-of-candy-basket/）binary_search
1482（https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/）binary_search
2528（https://leetcode.com/problems/maximize-the-minimum-powered-city/description/）binary_search|prefix_sum|diff_array|greedy
2560（https://leetcode.com/problems/house-robber-iv/）binary_search|dp
2234（https://leetcode.com/problems/maximum-total-beauty-of-the-gardens/description/）prefix_sum|binary_search|brute_force

=====================================LuoGu======================================
P1577（https://www.luogu.com.cn/problem/P1577）math|floor|binary_search
P1570（https://www.luogu.com.cn/problem/P1570）math|greedy|binary_search
P1843（https://www.luogu.com.cn/problem/P1843）greedy|binary_search
P2309（https://www.luogu.com.cn/problem/P2309）prefix_sum|sorted_list|binary_search|counter|sub_consequence_sum
P2390（https://www.luogu.com.cn/problem/P2390）brute_force|binary_search|two_pointers
P2759（https://www.luogu.com.cn/problem/P2759）math|binary_search
P1404（https://www.luogu.com.cn/problem/P1404）math|prefix_sum|binary_search
P1592（https://www.luogu.com.cn/problem/P1592）binary_search|inclusion_exclusion|kth_coprime_of_n
P2855（https://www.luogu.com.cn/problem/P2855）greedy|binary_search
P2884（https://www.luogu.com.cn/problem/P2884）binary_search
P2985（https://www.luogu.com.cn/problem/P2985）greedy|binary_search|implemention
P3184（https://www.luogu.com.cn/problem/P3184）binary_search|counter
P3611（https://www.luogu.com.cn/problem/P3611）binary_search|greedy|heapq|implemention
P3743（https://www.luogu.com.cn/problem/P3743）binary_search
P4058（https://www.luogu.com.cn/problem/P4058）binary_search
P4670（https://www.luogu.com.cn/problem/P4670）sorting|binary_search|counter
P5119（https://www.luogu.com.cn/problem/P5119）greedy|binary_search
P5250（https://www.luogu.com.cn/problem/P5250）sorted_list
P6174（https://www.luogu.com.cn/problem/P6174）greedy|binary_search
P6281（https://www.luogu.com.cn/problem/P6281）greedy|binary_search
P6423（https://www.luogu.com.cn/problem/P6423）binary_search
P7177（https://www.luogu.com.cn/problem/P7177）binary_search|tree|dfs|implemention
P1314（https://www.luogu.com.cn/problem/P1314）binary_search|sum_nearest_subsequence
P3017（https://www.luogu.com.cn/problem/P3017）binary_search|sub_matrix_sum|max_min
P1083（https://www.luogu.com.cn/problem/P1083）binary_search|diff_array
P1281（https://www.luogu.com.cn/problem/P1281）binary_search|specific_plans
P1381（https://www.luogu.com.cn/problem/P1381）binary_search|sliding_window|brain_teaser
P1419（https://www.luogu.com.cn/problem/P1419）binary_search|priority_queue
P1525（https://www.luogu.com.cn/problem/P1525）binary_search|bfs|bipartite_graph|unionfind|coloring
P1542（https://www.luogu.com.cn/problem/P1542）binary_search|fraction|high_precision
P2237（https://www.luogu.com.cn/problem/P2237）brain_teaser|sorting|binary_search
P2810（https://www.luogu.com.cn/problem/P2810）binary_search|brute_force
P3718（https://www.luogu.com.cn/problem/P3718）binary_search|greedy
P3853（https://www.luogu.com.cn/problem/P3853）binary_search|greedy
P4343（https://www.luogu.com.cn/problem/P4343）bound|binary_search|implemention
P5844（https://www.luogu.com.cn/problem/P5844）median|greedy|prefix_sum|binary_search
P5878（https://www.luogu.com.cn/problem/P5878）binary_search|brute_force
P6004（https://www.luogu.com.cn/problem/P6004）binary_search|union_find
P6058（https://www.luogu.com.cn/problem/P6058）dfs_order|offline_lca|binary_search
P6069（https://www.luogu.com.cn/problem/P6069）math|binary_search
P6733（https://www.luogu.com.cn/problem/P6733）binary_search|sorted_list
P8161（https://www.luogu.com.cn/problem/P8161）greedy|binary_search
P8198（https://www.luogu.com.cn/problem/P8198）binary_search|pointer
P9050（https://www.luogu.com.cn/problem/P9050）binary_search|data_range|greedy|implemention

===================================CodeForces===================================
1251D（https://codeforces.com/problemset/problem/1251/D）greedy|median|binary_search
830A（https://codeforces.com/problemset/problem/830/A）greedy|point_coverage|binary_search
847E（https://codeforces.com/problemset/problem/847/E）greedy|binary_search|pointer
732D（https://codeforces.com/problemset/problem/732/D）greedy|binary_search|pointer
778A（https://codeforces.com/problemset/problem/778/A）binary_search|pointer
913C（https://codeforces.com/problemset/problem/913/C）dp|binary_search|greedy|implemention
1791G2（https://codeforces.com/problemset/problem/1791/G2）greedy|sorting|prefix_sum|brute_force|binary_search
448D（https://codeforces.com/problemset/problem/448/D）binary_search|kth_max_of_n_mul_m_table
1475D（https://codeforces.com/problemset/problem/1475/D）greedy|sorting|prefix_sum|brute_force|binary_search
1370D（https://codeforces.com/problemset/problem/1370/D）binary_search|greedy|check
1486D（https://codeforces.com/problemset/problem/1486/D）binary_search|hash|prefix_sum|maximum_length_of_sub_consequence_with_pos_sum
1118D2（https://codeforces.com/problemset/problem/1118/D2）greedy|binary_search
883I（https://codeforces.com/problemset/problem/883/I）binary_search|two_pointers|dp
1538G（https://codeforces.com/contest/1538/problem/G）binary_search|brute_force|math
1680C（https://codeforces.com/contest/1680/problem/C）binary_search|greedy|two_pointers|check
1251D（https://codeforces.com/contest/1251/problem/D）greedy|binary_search

====================================AtCoder=====================================
D - No Need （https://atcoder.jp/contests/abc056/tasks/arc070_b）binary_search|bag_dp
D - Widespread（https://atcoder.jp/contests/abc063/tasks/arc075_b）binary_search|greedy

=====================================AcWing=====================================
120（https://www.acwing.com/problem/content/122/）binary_search
14（https://www.acwing.com/problem/content/description/15/）pigeonhole_principle|binary_search
3973（https://www.acwing.com/problem/content/3976/）high_precision|binary_search|sliding_window|two_pointers
4863（https://www.acwing.com/problem/content/description/4866/）binary_search|pigeonhole_principle
5048（https://www.acwing.com/problem/content/description/5051/）high_precision|binary_search|specific_plan

"""
import bisect
import math
from collections import deque, defaultdict
from itertools import accumulate, combinations
from math import inf
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.data_structure.sorted_list.template import LocalSortedList
from src.graph.tree_lca.template import OfflineLCA
from src.graph.union_find.template import UnionFind
from src.mathmatics.number_theory.template import NumberTheory
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1314(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1314
        tag: binary_search|sum_nearest_subsequence
        """

        # binary_search寻找最接近目标值的和
        n, m, s = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        queries = [ac.read_list_ints() for _ in range(m)]

        def check(w):
            cnt = [0] * (n + 1)
            pre = [0] * (n + 1)
            for i in range(n):
                cnt[i + 1] = cnt[i] + int(nums[i][0] >= w)
                pre[i + 1] = pre[i] + int(nums[i][0] >= w) * nums[i][1]
            res = 0
            for a, b in queries:
                res += (pre[b] - pre[a - 1]) * (cnt[b] - cnt[a - 1])
            return res

        ans = inf
        low = 0
        high = max(ls[0] for ls in nums)
        while low < high - 1:
            mid = low + (high - low) // 2
            x = check(mid)
            ans = ac.min(ans, abs(s - x))
            if x <= s:
                high = mid - 1
            else:
                low = mid + 1
        ans = ac.min(ans, ac.min(abs(s - check(low)), abs(s - check(high))))
        ac.st(ans)
        return

    @staticmethod
    def cf_448d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/448/D
        tag: binary_search|kth_max_of_n_mul_m_table
        """
        #  n*m 乘法矩阵内的第 k 大元素
        n, m, k = ac.read_list_ints()

        def check(num):
            res = 0
            for x in range(1, m + 1):
                res += min(n, num // x)
            return res >= k

        # 初始化大小
        if m > n:
            m, n = n, m

        ans = BinarySearch().find_int_left(1, m * n, check)
        ac.st(ans)
        return

    @staticmethod
    def cf_1370d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1370/D
        tag: binary_search|greedy|check
        """
        n, k = map(int, input().split())
        nums = list(map(int, input().split()))

        def check(x):
            # 奇偶交替，依次brute_force奇数索引与偶数索引不超过x
            for ind in [0, 1]:
                cnt = 0
                for num in nums:
                    if not ind:
                        cnt += 1
                        ind = 1 - ind
                    else:
                        if num <= x:
                            cnt += 1
                            ind = 1 - ind
                    if cnt >= k:
                        return True
            return False

        low = min(nums)
        high = max(nums)
        ans = BinarySearch().find_int_left(low, high, check)
        ac.st(ans)
        return

    @staticmethod
    def cf_1475d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1475/D
        tag: greedy|sorting|prefix_sum|brute_force|binary_search
        """
        # greedysorting后，brute_force并prefix_sumbinary_search查询
        for _ in range(ac.read_int()):
            n, m = ac.read_list_ints()
            a = ac.read_list_ints()
            b = ac.read_list_ints()
            if sum(a) < m:
                ac.st(-1)
                continue

            # sorting
            a1 = [a[i] for i in range(n) if b[i] == 1]
            a2 = [a[i] for i in range(n) if b[i] == 2]
            a1.sort(reverse=True)
            a2.sort(reverse=True)

            # prefix_sum
            x, y = len(a1), len(a2)
            pre1 = [0] * (x + 1)
            for i in range(x):
                pre1[i + 1] = pre1[i] + a1[i]

            # 初始化后binary_searchbrute_force
            ans = inf
            pre = 0
            j = bisect.bisect_left(pre1, m - pre)
            if j <= x:
                ans = ac.min(ans, j)

            for i in range(y):
                cnt = i + 1
                pre += a2[i]
                j = bisect.bisect_left(pre1, m - pre)
                if j <= x:
                    ans = ac.min(ans, j + cnt * 2)
            ac.st(ans)
        return

    @staticmethod
    def lg_p3017(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3017
        tag: binary_search|sub_matrix_sum|max_min
        """

        # binary_search将矩阵分成a*b个子矩阵且子矩阵和的最小值最大
        def check(x):

            def cut():
                cur = 0
                c = 0
                for num in pre:
                    cur += num
                    if cur >= x:
                        c += 1
                        cur = 0
                return c >= b

            cnt = i = 0
            pre = [0] * n
            while i < m:
                if cut():
                    pre = [0] * n
                    cnt += 1
                else:
                    for j in range(n):
                        pre[j] += grid[i][j]
                    i += 1
            if cut():
                cnt += 1
            return cnt >= a

        m, n, a, b = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]
        low = 0
        high = sum(sum(g) for g in grid) // (a * b)
        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        ans = high if check(high) else low
        ac.st(ans)
        return

    @staticmethod
    def cf_1680c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1680/problem/C
        tag: binary_search|greedy|two_pointers|check
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            n = len(s)
            tot_1 = s.count("1")

            def check(x):
                j = cnt = ceil_1 = 0
                cnt_1 = 0
                for i in range(n):
                    while j < n and (cnt + int(s[j] == "0")) <= x:
                        cnt += s[j] == "0"
                        cnt_1 += s[j] == "1"
                        j += 1
                    if cnt <= x:
                        ceil_1 = ac.max(ceil_1, cnt_1)
                    if s[i] == "0":
                        cnt -= 1
                    else:
                        cnt_1 -= 1
                return tot_1 - ceil_1 <= x

            ac.st(BinarySearch().find_int_left(0, n, check))
        return

    @staticmethod
    def cf_1791g2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1791/G2
        tag: greedy|sorting|prefix_sum|brute_force|binary_search
        """
        # find_int_right
        for _ in range(ac.read_int()):

            n, c = ac.read_list_ints()
            cost = ac.read_list_ints()
            lst = [[ac.min(x, n + 1 - x) + cost[x - 1], x + cost[x - 1]]
                   for x in range(1, n + 1)]
            lst.sort(key=lambda it: it[0])
            pre = [0] * (n + 1)
            for i in range(n):
                pre[i + 1] = pre[i] + lst[i][0]

            def check(y):
                res = pre[y]
                if y > i:
                    res -= lst[i][0]
                res += lst[i][1]
                return res <= c

            ans = 0
            for i in range(n):
                if lst[i][1] <= c:
                    cur = BinarySearch().find_int_right(0, n, check)
                    cur = cur - int(cur > i) + 1
                    ans = ac.max(ans, cur)
            ac.st(ans)
        return

    @staticmethod
    def lc_1889(packages: List[int], boxes: List[List[int]]) -> int:
        """
        url: https://leetcode.com/problems/minimum-space-wasted-from-packaging/
        tag: sorting|prefix_sum|greedy|binary_search
        """
        # sorting|prefix_sumpreprocess与greedybinary_search
        packages.sort()
        pre = list(accumulate(packages, initial=0))
        n = len(packages)
        ans = inf
        mod = 10 ** 9 + 7
        for box in boxes:
            box.sort()
            if box[-1] < packages[-1]:
                continue
            i = cur = 0
            for num in box:
                if i == n:
                    break
                if num < packages[i]:
                    continue

                j = bisect.bisect_right(packages, num) - 1
                cur += (j - i + 1) * num - (pre[j + 1] - pre[i])
                i = j + 1
            if cur < ans:
                ans = cur
        return ans % mod if ans < inf else -1

    @staticmethod
    def lc_2141(n: int, batteries: List[int]) -> int:
        """
        url: https://leetcode.com/problems/maximum-running-time-of-n-computers/
        tag: greedy|binary_search|implemention
        """
        # greedybinary_search

        batteries.sort(reverse=True)
        rest = sum(batteries[n:])

        def check(w):
            res = 0
            for num in batteries[:n]:
                if num < w:
                    res += w - num
            return res <= rest

        return BinarySearch().find_int_right(0, batteries[n - 1] + rest, check)

    @staticmethod
    def lc_2528(stations: List[int], r: int, k: int) -> int:
        """
        url: https://leetcode.com/problems/maximize-the-minimum-powered-city/description/
        tag: binary_search|prefix_sum|diff_array|greedy
        """
        # binary_searchprefix_sumdiff_array|greedy验证
        n = len(stations)
        nums = [0] * n
        for i in range(n):
            left = max(0, i - r)
            nums[left] += stations[i]
            if i + r + 1 < n:
                nums[i + r + 1] -= stations[i]
        for i in range(1, n):
            nums[i] += nums[i - 1]

        def check(x):
            diff = [0] * (n + 2 * r + 10)
            res = 0
            for j in range(n):
                diff[j] += diff[j - 1] if j else 0
                cur = diff[j] + nums[j]
                if cur < x:
                    res += x - cur
                    diff[j] += x - cur
                    diff[j + 2 * r + 1] -= x - cur

            return res <= k

        return BinarySearch().find_int_right(0, max(nums) + k, check)

    @staticmethod
    def lc_2563(nums, lower, upper):
        """
        url: https://leetcode.com/problems/count-the-number-of-fair-pairs/
        tag: binary_search|sorted_list
        """
        # 查找数值和在一定范围的数对个数
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n):
            x = bisect.bisect_right(nums, upper - nums[i], hi=i)
            y = bisect.bisect_left(nums, lower - nums[i], hi=i)
            ans += x - y
        return ans

    @staticmethod
    def lc_4(nums1: List[int], nums2: List[int]) -> float:
        """
        url: https://leetcode.com/problems/median-of-two-sorted-arrays/
        tag: binary_search|median|two_arrays|same_direction_pointer
        """
        # two_pointersbinary_search移动查找两个正序数组的median
        def get_kth_num(k):
            ind1 = ind2 = 0
            while k:
                if ind1 == m:
                    return nums2[ind2 + k - 1]
                if ind2 == n:
                    return nums1[ind1 + k - 1]
                if k == 1:
                    return min(nums1[ind1], nums2[ind2])
                index1 = min(ind1 + k // 2 - 1, m - 1)
                index2 = min(ind2 + k // 2 - 1, n - 1)
                val1 = nums1[index1]
                val2 = nums2[index2]
                if val1 < val2:
                    k -= index1 - ind1 + 1
                    ind1 = index1 + 1
                else:
                    k -= index2 - ind2 + 1
                    ind2 = index2 + 1
            return

        m, n = len(nums1), len(nums2)
        s = m + n
        if s % 2:
            return get_kth_num(s // 2 + 1)
        else:
            return (get_kth_num(s // 2 + 1) + get_kth_num(s // 2)) / 2

    @staticmethod
    def cf_1486d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1486/D
        tag: binary_search|hash|prefix_sum|maximum_length_of_sub_consequence_with_pos_sum
        """
        # 转换为binary_search和hash前缀求最长和为正数的最长连续子序列
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        low = 0
        high = n - 1
        lst = sorted(nums)

        def check(x):
            x = lst[x]
            dct = dict()
            pre = 0
            dct[0] = -1
            for i, num in enumerate(nums):
                pre += 1 if num >= x else -1
                if pre > 0 and i + 1 >= k:
                    return True
                # 为负数时，只需greedy考虑第一次为 pre-1 时的长度即可
                if pre - 1 in dct and i - dct[pre - 1] >= k:
                    return True
                if pre not in dct:
                    dct[pre] = i
            return False

        while low < high - 1:
            mid = low + (high - low) // 2
            if check(mid):
                low = mid
            else:
                high = mid
        ans = high if check(high) else low
        ac.st(lst[ans])
        return

    @staticmethod
    def lg_p1083(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1083
        tag: binary_search|diff_array
        """

        # binary_search结合差分寻找第一个失效点

        def check(s):
            diff = [0] * n
            for c, a, b in lst[:s]:
                diff[a - 1] += c
                if b < n:
                    diff[b] -= c
            if diff[0] > nums[0]:
                return False
            pre = diff[0]
            for i in range(1, n):
                pre += diff[i]
                if pre > nums[i]:
                    return False
            return True

        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        lst = [ac.read_list_ints() for _ in range(m)]
        ans = BinarySearch().find_int_right(0, n, check)
        if ans == n:
            ac.st(0)
        else:
            ac.st(-1)
            ac.st(ans + 1)
        return

    @staticmethod
    def ac_120(ac=FastIO()):

        def check(pos):
            res = 0
            for s, e, d in nums:
                if s <= pos:
                    res += (ac.min(pos, e) - s) // d + 1
            return res % 2 == 1

        def compute(pos):
            res = 0
            for s, e, d in nums:
                if s <= pos <= e:
                    res += (pos - s) % d == 0
            return [pos, res]

        # binary_search
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = [ac.read_list_ints() for _ in range(n)]
            low = min(x for x, _, _ in nums)
            high = max(x for _, x, _ in nums)
            while low < high - 1:
                mid = low + (high - low) // 2
                if check(mid):
                    high = mid
                else:
                    low = mid
            if check(low):
                ac.lst(compute(low))
            elif check(high):
                ac.lst(compute(high))
            else:
                ac.st("There's no weakness.")
        return

    @staticmethod
    def abc_56d(ac=FastIO()):
        # binary_search，用bag_dp|check
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        nums.sort()

        def check(i):
            dp = [0] * k
            dp[0] = 1
            xx = nums[i]
            if xx >= k:
                return False

            for j in range(n):
                if j != i:
                    x = nums[j]
                    for p in range(k - 1, x - 1, -1):
                        if dp[p - x]:
                            dp[p] = 1
                            if p + xx >= k:
                                return False  # 此时为必要

            return True  # 为非必要的目标元素

        ans = BinarySearch().find_int_right(0, n - 1, check)  # 非必要具有单调性，更小的也为非必要
        if check(ans):
            ac.st(ans + 1)
        else:
            ac.st(0)
        return

    @staticmethod
    def abc_63d(ac=FastIO()):
        # binary_search，greedycheck
        n, a, b = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]

        def check(s):
            res = 0
            for num in nums:
                if num > s * b:
                    res += ac.ceil((num - s * b), (a - b))
            return res <= s

        ans = BinarySearch().find_int_left(0, ac.ceil(max(nums), b), check)
        ac.st(ans)
        return

    @staticmethod
    def ac_14(nums):
        # 利用pigeonhole_principlebinary_search
        n = len(nums) - 1
        low = 1
        high = n
        while low < high:
            mid = low + (high - low) // 2
            cnt = 0
            for num in nums:
                if low <= num <= mid:
                    cnt += 1
            if cnt > mid - low + 1:
                high = mid
            else:

                low = mid + 1
        return low

    @staticmethod
    def lg_p1281(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1281
        tag: binary_search|specific_plans
        """
        # classicalbinary_search并specific_plans
        m, k = ac.read_list_ints()
        nums = ac.read_list_ints()

        def check(xx):
            res = pp = 0
            for ii in range(m - 1, -1, -1):
                if pp + nums[ii] > xx:
                    res += 1
                    pp = nums[ii]
                else:
                    pp += nums[ii]
                if res + 1 > k:
                    return False
            return True

        x = BinarySearch().find_int_left(max(nums), sum(nums), check)
        ans = []
        pre = nums[m - 1]
        post = m - 1
        for i in range(m - 2, -1, -1):
            if pre + nums[i] > x:
                ans.append([i + 2, post + 1])
                pre = nums[i]
                post = i
            else:
                pre += nums[i]
        ans.append([1, post + 1])
        for a in ans[::-1]:
            ac.lst(a)
        return

    @staticmethod
    def lg_p1381(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1381
        tag: binary_search|sliding_window|brain_teaser
        """
        # classicalbinary_search
        n = ac.read_int()
        dct = set([ac.read_str() for _ in range(n)])
        m = ac.read_int()
        words = [ac.read_str() for _ in range(m)]
        cur = set()
        for w in words:
            if w in dct:
                cur.add(w)

        def check(x):
            cnt = defaultdict(int)
            cc = 0
            for i in range(m):
                # sliding_window判断是否可行
                if words[i] in dct:
                    cnt[words[i]] += 1
                    if cnt[words[i]] == 1:
                        cc += 1
                        if cc == s:
                            return True
                if i >= x - 1:
                    if words[i - x + 1] in dct:
                        cnt[words[i - x + 1]] -= 1
                        if not cnt[words[i - x + 1]]:
                            cc -= 1
            return False

        # greedy选取所有能背的单词
        s = len(cur)
        ac.st(s)
        if not s:
            ac.st(0)
            return
        ans = BinarySearch().find_int_left(1, m, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1592(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1592
        tag: binary_search|inclusion_exclusion|kth_coprime_of_n
        """
        n, k = ac.read_list_ints()
        if n == 1:  # 特判
            ac.st(k)
            return
        lst = NumberTheory().get_prime_factor(n)
        prime = [x for x, _ in lst]
        m = len(prime)

        def check(x):
            # inclusion_exclusion与 n 不coprime且小于等于 x 的数个数
            res = 0
            for i in range(1, m + 1):
                for item in combinations(prime, i):
                    cur = 1
                    for num in item:
                        cur *= num
                    res += (x // cur) * (-1) ** (i + 1)
            return x - res >= k

        ans = BinarySearch().find_int_left(1, n * k, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1419(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1419
        tag: binary_search|priority_queue
        """

        # binary_search|priority_queue
        def check(x):
            stack = deque()
            res = []
            # 单调队列记录前序最小值
            for i in range(n):
                while stack and stack[0][0] <= i - k:
                    stack.popleft()
                while stack and stack[-1][1] >= pre[i] - x * i:
                    stack.pop()
                stack.append([i, pre[i] - x * i])
                res.append(stack[0][1])  # 记录长度在 k 左右的最小前缀变化和
                if i >= s - 1:
                    if pre[i + 1] - x * (i + 1) >= res[i - s + 1]:
                        return True
            return False

        n = ac.read_int()
        s, t = ac.read_list_ints()
        nums = []
        for _ in range(n):
            nums.append(int(input().strip()))
        pre = [0] * (n + 1)
        for j in range(n):
            pre[j + 1] = pre[j] + nums[j]

        # binary_search最大平均值
        k = t - s
        ans = BinarySearch().find_float_right(min(nums), max(nums), check)
        ac.st("%.3f" % ans)
        return

    @staticmethod
    def lg_p1525(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1525
        tag: binary_search|bfs|bipartite_graph|unionfind|coloring
        """
        # binary_search|bfsbipartite_graph划分
        n, m = ac.read_list_ints()
        lst = [ac.read_list_ints() for _ in range(m)]

        def check(weight):
            edges = [[i, j] for i, j, w in lst if w > weight]
            dct = defaultdict(list)
            for i, j in edges:
                dct[i].append(j)
                dct[j].append(i)
            # coloring_method判断是否可以binary_search
            visit = [0] * (n + 1)
            for i in range(1, n + 1):
                if visit[i] == 0:
                    stack = [i]
                    visit[i] = 1
                    order = 2
                    while stack:
                        nex = []
                        for j in stack:
                            for y in dct[j]:
                                if not visit[y]:
                                    visit[y] = order
                                    nex.append(y)
                        order = 1 if order == 2 else 2
                        stack = nex

            return all(visit[i] != visit[j] for i, j in edges)

        # binary_search最小的最大值
        low = 0
        high = max(ls[-1] for ls in lst)
        ans = BinarySearch().find_int_left(low, high, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1542(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1542
        tag: binary_search|fraction|high_precision
        """
        # binary_search|分数high_precision
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]

        def add(lst1, lst2):
            # 分数|减
            a, b = lst1
            c, d = lst2
            d1 = a * d + c * b
            d2 = b * d
            return [d1, d2]

        def check(xx):
            # 最早与最晚出发
            res = [xx, 1]
            while int(res[0]) != res[0]:
                res[0] *= 10
                res[1] *= 10
            res = [int(w) for w in res]
            t1 = [0, 1]
            for x, y, s in nums:
                cur = add(t1, [s * res[1], res[0]])
                if cur[0] > y * cur[1]:
                    return False
                t1 = cur[:]
                if cur[0] < x * cur[1]:
                    t1 = [x, 1]
            return True

        ans = BinarySearch().find_float_left(1e-4, 10 ** 7, check)
        ac.st("%.2f" % ans)
        return

    @staticmethod
    def cf_1118d2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1118/D2
        tag: greedy|binary_search
        """

        # greedybinary_search
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        s = sum(nums)
        if s < m:
            ac.st(-1)
            return
        nums.sort(reverse=True)

        def check(x):
            ans = 0
            for i in range(n):
                j = i // x
                ans += ac.max(0, nums[i] - j)
            return ans >= m

        ac.st(BinarySearch().find_int_left(1, n, check))
        return

    @staticmethod
    def cf_883i(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/883/I
        tag: binary_search|two_pointers|dp
        """
        # binary_search|two_pointersdp
        n, k = ac.read_list_ints()
        nums = sorted(ac.read_list_ints())

        def check(x):
            dp = [0] * (n + 1)
            dp[0] = 1
            j = 0
            for i in range(n):
                while nums[i] - nums[j] > x:
                    j += 1
                while not dp[j] and j < i - k + 1:
                    j += 1
                if dp[j] and i + 1 - j >= k:
                    dp[i + 1] = 1
            return dp[-1] == 1

        ans = BinarySearch().find_int_left(0, nums[-1] - nums[0], check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2237(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2237
        tag: brain_teaser|sorting|binary_search
        """
        # brain_teasersorting后binary_search
        w, n = ac.read_list_ints()
        nums = [ac.read_str() for _ in range(w)]
        ind = list(range(w))
        ind.sort(key=lambda it: nums[it])
        nums.sort()
        for _ in range(n):
            k, s = ac.read_list_strs()
            k = int(k)
            x = bisect.bisect_left(nums, s) + k - 1
            if x < w and nums[x][:len(s)] == s:
                ac.st(ind[x] + 1)
            else:
                ac.st(-1)
        return

    @staticmethod
    def lg_p2810(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2810
        tag: binary_search|brute_force
        """

        # binary_search|brute_force
        n = ac.read_int()

        low = 0
        high = 10 ** 18

        def check2(x):
            k = 2
            res = 0
            while k * k * k <= x:
                res += x // (k * k * k)
                k += 1
            return res

        def check(x):
            k = 2
            res = 0
            while k * k * k <= x:
                res += x // (k * k * k)
                k += 1
            return res >= n

        ans = BinarySearch().find_int_left(low, high, check)
        if check2(ans) == n:
            ac.st(ans)
            return
        ac.st(-1)
        return

    @staticmethod
    def lg_p3718(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3718
        tag: binary_search|greedy
        """
        # binary_search|greedy
        n, k = ac.read_list_ints()
        s = ac.read_str()

        def check(x):
            if x == 1:
                # 特殊情况
                op1 = op2 = 0
                for i in range(n):
                    if i % 2:
                        op1 += 1 if s[i] == "N" else 0
                        op2 += 1 if s[i] == "F" else 0
                    else:
                        op1 += 1 if s[i] == "F" else 0
                        op2 += 1 if s[i] == "N" else 0
                return op1 <= k or op2 <= k

            op = 0
            pre = s[0]
            cnt = 1
            for w in s[1:]:
                if w == pre:
                    cnt += 1
                else:
                    # 对于相同状态连续区间断开的最少操作次数
                    op += cnt // (x + 1)
                    pre = w
                    cnt = 1
            op += cnt // (x + 1)
            return op <= k

        ans = BinarySearch().find_int_left(1, n, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p3853(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3853
        tag: binary_search|greedy
        """
        # binary_searchgreedy题
        length, n, k = ac.read_list_ints()
        lst = ac.read_list_ints()
        lst.sort()

        def check(x):
            return sum((lst[i + 1] - lst[i] + x - 1) // x - 1 for i in range(n - 1)) <= k

        low = 1
        high = max(lst[i + 1] - lst[i] for i in range(n - 1))
        ans = BinarySearch().find_int_left(low, high, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p4343(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4343
        tag: bound|binary_search|implemention
        """
        # 上下界binary_search|implemention
        l, k = ac.read_list_ints()
        lst = []
        for _ in range(l):
            lst.append(int(input().strip()))
        low = 1
        high = sum(abs(ls) for ls in lst)

        def compute(n):
            cnt = cur = 0
            for num in lst:
                cur += num
                if cur >= n:
                    cnt += 1
                    cur = 0
                cur = 0 if cur < 0 else cur
            return cnt

        def check1(n):
            return compute(n) >= k

        def check2(n):
            return compute(n) <= k

        # 的binary_search函数写法与出specific_plan
        ceil = BinarySearch().find_int_right(low, high, check1)
        if compute(ceil) != k:
            ac.st(-1)
            return
        floor = BinarySearch().find_int_left(low, high, check2)
        ac.lst([floor, ceil])
        return

    @staticmethod
    def lg_p5844(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5844
        tag: median|greedy|prefix_sum|binary_search
        """
        # mediangreedy与prefix_sumbinary_search
        n, m, b = ac.read_list_ints()
        pos = [ac.read_int() for _ in range(n)]
        ans = j = 0
        pre = ac.accumulate(pos)

        def check(x, y):
            mid = (x + y) // 2
            left = (mid - x) * pos[mid] - (pre[mid] - pre[x])
            right = pre[y + 1] - pre[mid + 1] - (y - mid) * pos[mid]
            return left + right

        for i in range(n):
            # brute_force左端点binary_search右端点
            while j < n and check(i, j) <= b:
                j += 1
            ans = ac.max(ans, j - i)
        ac.st(ans)
        return

    @staticmethod
    def lg_p5878(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5878
        tag: binary_search|brute_force
        """
        # binary_search|brute_force
        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]

        def check(num):
            cost = 0
            for x, y, sm, pm, sv, pv in nums:
                need = num * x - y
                if need <= 0:
                    continue
                cur = inf
                # brute_force小包装个数
                for i in range(need + 1):
                    rest = need - i * sm
                    if rest > 0:
                        cur = ac.min(cur, i * pm + math.ceil(rest / sv) * pv)
                    else:
                        cur = ac.min(cur, i * pm)
                        break
                cost += cur
                if cost > m:
                    return False

            return cost <= m

        ans = BinarySearch().find_int_right(0, m, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6004(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6004
        tag: binary_search|union_find
        """
        # binary_search|union_find
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints_minus_one()
        edges = [ac.read_list_ints() for _ in range(m)]
        edges.sort(key=lambda it: -it[2])

        def check(x):
            uf = UnionFind(n)
            for i, j, _ in edges[:x]:
                uf.union(i - 1, j - 1)
            group = uf.get_root_part()
            for g in group:
                cur = set([nums[i] for i in group[g]])
                if not all(i in cur for i in group[g]):
                    return False
            return True

        ans = BinarySearch().find_int_left(0, m, check)
        ac.st(-1 if not ans else edges[ans - 1][2])
        return

    @staticmethod
    def lg_p6058(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6058
        tag: dfs_order|offline_lca|binary_search
        """
        # dfs_order与offline_lca 相邻叶子之间距离并binary_search确定时间
        n, k = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            x, y, z = ac.read_list_ints_minus_one()
            dct[x].append([y, z + 1])
            dct[y].append([x, z + 1])
        if n == 1:
            ac.st(0)
            return
        for i in range(n):
            dct[i].sort(reverse=True)
        # 找出叶子
        stack = [[0, -1]]
        dis = [0] * n
        leaf = []
        while stack:
            i, fa = stack.pop()
            for j, w in dct[i]:
                if j != fa:
                    dis[j] = dis[i] + w
                    stack.append([j, i])
            if len(dct[i]) == 1 and i:
                leaf.append(i)
        c = len(leaf)
        pairs = [[leaf[i - 1], leaf[i]] for i in range(1, c)]
        edge = [[ls[0] for ls in lst] for lst in dct]
        # 叶子之间的距离
        ces = OfflineLCA().bfs_iteration(edge, pairs, 0)
        pairs_dis = [dis[leaf[i - 1]] + dis[leaf[i]] - 2 * dis[ces[i - 1]] for i in range(1, c)]
        pre = ac.accumulate(pairs_dis)

        def check(t):
            ii = 0
            part = 0
            while ii < c:
                post = -1
                for jj in range(ii, c):
                    # 当前节点最远能够到达的叶子距离
                    if pre[jj] - pre[ii] + dis[leaf[ii]] + dis[leaf[jj]] <= t:
                        post = jj
                    else:
                        break
                part += 1
                ii = post + 1
            return part <= k

        # binary_search
        ans = BinarySearch().find_int_left(max(dis[i] * 2 for i in leaf), sum(dis[i] * 2 for i in leaf), check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p6069(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6069
        tag: math|binary_search
        """
        # 方差公式变形，binary_search|变量维护区间的方差值大小
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        nums.sort()

        def check(x):
            ss = s = 0
            for i in range(n):
                ss += nums[i] ** 2
                s += nums[i]
                if i >= x - 1:
                    # 方差变形公式转换为整数乘法
                    if x * ss - s * s <= x * m:
                        return True
                    ss -= nums[i - x + 1] ** 2
                    s -= nums[i - x + 1]
            return False

        ans = BinarySearch().find_int_right(1, n, check)
        ac.st(n - ans)
        return

    @staticmethod
    def lg_p6633(ac=FastIO()):
        # binary_search|STL Check
        n, k = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: -it[1])

        def check(x):
            res = 0
            pre = LocalSortedList()
            for a, c in nums:
                res += pre.bisect_right(a * c - x * a)
                pre.add(-(a * c - x * a))
                if res >= k:
                    return True
            return res >= k

        ans = BinarySearch().find_float_right(0, nums[0][1], check, 1e-3)
        ac.st(ans)
        return

    @staticmethod
    def lc_2594(ranks: List[int], cars: int) -> int:
        """
        url: https://leetcode.com/problems/minimum-time-to-repair-cars/
        tag: binary_search
        """
        #  binary_search

        def check(x):
            res = 0
            for r in ranks:
                res += int((x / r) ** 0.5)
            return res >= cars

        return BinarySearch().find_int_left(0, ranks[0] * cars ** 2, check)

    @staticmethod
    def lc_2604(hens: List[int], grains: List[int]) -> int:
        """
        url: https://leetcode.com/problems/minimum-time-to-eat-all-grains/
        tag: binary_search|greedy|pointer
        """
        # binary_search|pointergreedy check
        hens.sort()
        grains.sort()
        m, n = len(hens), len(grains)

        def check(x):
            i = 0
            for pos in hens:
                left = right = 0
                while i < n:
                    if grains[i] >= pos:
                        right = right if right > grains[i] - pos else grains[i] - pos
                    else:
                        left = left if left > pos - grains[i] else pos - grains[i]
                    if left * 2 + right <= x or right * 2 + left <= x:
                        i += 1
                    else:
                        break
                if i == n:
                    return True
            return False

        low = 0
        high = sum(abs(g - hens[0]) * 2 for g in grains)
        return BinarySearch().find_int_left(low, high, check)

    @staticmethod
    def lg_p8161(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8161
        tag: greedy|binary_search
        """
        # greedy|binary_search
        n, m = ac.read_list_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()

        def check(x):
            res = 0
            for i in range(n):
                if a[i] < b[i]:
                    res += (x + b[i] - 1) // b[i]
                else:
                    if m * a[i] >= x:
                        res += (x + a[i] - 1) // a[i]
                    else:
                        res += m
                        res += (x - a[i] * m + b[i] - 1) // b[i]
                if res > m * n:
                    return False
            return res <= m * n

        low = 0
        high = 10 ** 18
        ans = BinarySearch().find_int_right(low, high, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8198(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8198
        tag: binary_search|pointer
        """
        # binary_search|pointer
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()

        def check(x):
            res = pre = 0
            for num in nums:
                if pre + num * num > x:
                    res += 1
                    pre = num * num
                else:
                    pre += num * num
            res += 1
            return res <= k

        low = max(nums) ** 2
        high = sum(num * num for num in nums)
        ans = BinarySearch().find_int_left(low, high, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p9050(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9050
        tag: binary_search|data_range|greedy|implemention
        """
        # binary_search注意data_range区间与greedyimplemention
        n = ac.read_int()
        if n == 1:
            ac.st("T")
            return

        nums = ac.read_list_ints()
        lst = sorted(nums)

        def check(x):
            flag = 1
            res = lst[x]
            for w in lst:
                if w == lst[x] and flag:
                    flag = 0
                    continue
                if res <= w:
                    return False
                res += w
            return True

        floor = BinarySearch().find_int_left(0, n - 1, check)
        if check(floor):
            ans = ["N" if nums[x] < lst[floor] else "T" for x in range(n)]
            ac.st("".join(ans))
        else:
            ac.st("N" * n)
        return

    @staticmethod
    def ac_3973(ac=FastIO()):
        # 浮点数binary_search与sliding_windowtwo_pointers
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        pos = ac.read_list_ints()
        pos.sort()
        nums.sort()

        def check(x):
            i = 0
            for num in nums:
                while i < m and not (pos[i] - x <= num <= pos[i] + x):
                    i += 1
                if i == m:
                    return False
            return True

        ans = BinarySearch().find_float_left(0, 2 * 10 ** 9, check)
        if ans - int(ans) >= 0.5:
            ac.st(int(ans) + 1)
        else:
            ac.st(int(ans))
        return

    @staticmethod
    def ac_4683(ac=FastIO()):
        # binary_search|pigeonhole_principle
        for _ in range(ac.read_int()):
            ac.read_str()
            m, n = ac.read_list_ints()
            grid = [ac.read_list_ints() for _ in range(m)]

            def check(x):
                row = [0] * m
                col = [0] * n
                for i in range(m):
                    for j in range(n):
                        if grid[i][j] >= x:
                            row[i] += 1
                            col[j] = 1
                if any(x == 0 for x in col):
                    return False
                if m <= n - 1:
                    return True
                return max(row) >= 2

            ac.st(BinarySearch().find_int_right(0, 10 ** 9, check))
        return

    @staticmethod
    def ac_5048(ac=FastIO()):
        # 浮点数binary_search并求出specific_plan
        ac.read_int()
        nums = ac.read_list_ints()
        nums.sort()

        def compute(r):
            pre = -inf
            res = []
            for num in nums:
                if num > pre:
                    res.append(num + r)
                    pre = num + 2 * r
                    if len(res) > 3:
                        break
            return res

        def check(r):
            return len(compute(r)) <= 3

        x = BinarySearch().find_float_left(0, nums[-1] - nums[0], check, 1e-6)
        ac.st(x)
        ans = compute(x)
        while len(ans) < 3:
            ans.append(ans[-1] + 1)
        ac.lst(ans)
        return