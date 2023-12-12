"""
Algorithm：segment_tree|segment_tree_binary_search
Description：range_sum|range_min|range_add|range_change|range_max|dynamic_segment_tree|defaultdict

====================================LeetCode====================================
218（https://leetcode.com/problems/the-skyline-problem/solution/by-liupengsay-isfo/）segment_tree|RangeChangeRangeMax
2286（https://leetcode.com/problems/booking-concert-tickets-in-groups/）segment_tree|RangeAddRangeSumMaxMin
2407（https://leetcode.com/problems/longest-increasing-subsequence-ii/）segment_tree|RangeAddRangeMax|linear_dp
2158（https://leetcode.com/problems/amount-of-new-area-painted-each-day/）segment_tree|RangeAddRangeSum
6318（https://leetcode.com/contest/weekly-contest-336/problems/minimum-time-to-complete-all-tasks/）segment_tree|greedy|binary_search
732（https://leetcode.com/problems/my-calendar-iii/）dynamic_segment_tree
1851（https://leetcode.com/problems/minimum-interval-to-include-each-query/）segment_tree|RangeChangeRangeMin|offline_query|priority_queue
2213（https://leetcode.com/problems/longest-substring-of-one-repeating-character/）segment_tree|sub_consequence|range_query|range_merge
2276（https://leetcode.com/problems/count-integers-in-intervals/）dynamic_segment_tree|union_find_range|SortedList
1340（https://leetcode.com/problems/jump-game-v/）segment_tree|linear_dp
2940（https://leetcode.com/problems/find-building-where-alice-and-bob-can-meet/）segment_tree_binary_search

=====================================LuoGu======================================
2846（https://www.luogu.com.cn/problem/P2846）segment_tree|range_reverse|range_sum
2574（https://www.luogu.com.cn/problem/P2574）segment_tree|range_reverse|range_sum
3130（https://www.luogu.com.cn/problem/P3130）RangeAddRangeSumMaxMin
3870（https://www.luogu.com.cn/problem/P3870）segment_tree|range_reverse|range_sum
5057（https://www.luogu.com.cn/problem/P5057）segment_tree|range_reverse|range_sum
3372（https://www.luogu.com.cn/problem/P3372）RangeAddRangeSumMaxMin
2880（https://www.luogu.com.cn/problem/P2880）RangeAddRangeSumMaxMin
1904（https://www.luogu.com.cn/problem/P1904）segment_tree|RangeAscendRangeMax
1438（https://www.luogu.com.cn/problem/P1438）diff_array|RangeAddRangeSumMaxMin|segment_tree
1253（https://www.luogu.com.cn/problem/P1253）range_add|range_change|segment_tree|range_sum
3373（https://www.luogu.com.cn/problem/P3373）range_add|range_mul|segment_tree|range_sum|RangeAddMulRangeSum
4513（https://www.luogu.com.cn/problem/P4513）segment_tree|range_change|range_merge|sub_consequence
1471（https://www.luogu.com.cn/problem/P1471）math|segment_tree|RangeAddRangeSum
6492（https://www.luogu.com.cn/problem/P6492）segment_tree|range_change|range_merge|sub_consequence
4145（https://www.luogu.com.cn/problem/P4145）math|segment_tree|RangeAddRangeSum
1558（https://www.luogu.com.cn/problem/P1558）segment_tree|RangeChangeRangeOr
3740（https://www.luogu.com.cn/problem/P3740）discretization|segment_tree|RangeChangeRangeSum
4588（https://www.luogu.com.cn/problem/P4588）segment_tree|RangeChangeRangeMul
6627（https://www.luogu.com.cn/problem/P6627）segment_tree|range_xor
8081（https://www.luogu.com.cn/problem/P8081）diff_array|counter|action_scop|segment_tree|RangeChangeRangeOr
8812（https://www.luogu.com.cn/problem/P8812）segment_tree|RangeDescendRangeMin
8856（https://www.luogu.com.cn/problem/solution/P8856）segment_tree|RangeAddRangeSumMaxMin

===================================CodeForces===================================
482B（https://codeforces.com/problemset/problem/482/B）segment_tree|RangeOrRangeAnd
380C（https://codeforces.com/problemset/problem/380/C）segment_tree|range_merge|sub_consequence|bracket
52C（https://codeforces.com/problemset/problem/52/C）segment_tree|circular_array|RangeChangeRangeMin
438D（https://codeforces.com/problemset/problem/438/D）segment_tree|range_sum|mod|RangeChangeRangeSumMaxMin
558E（https://codeforces.com/contest/558/problem/E）alphabet|segment_tree|sorting
343D（https://codeforces.com/problemset/problem/343/D）dfs_order|segment_tree
242E（https://codeforces.com/problemset/problem/242/E）segment_tree|RangeXorRangeOr
987C（https://codeforces.com/problemset/problem/987/C）brute_force|segment_tree|prefix_suffix
1216F（https://codeforces.com/contest/1216/problem/F）segment_tree|dp|monotonic_queue
1665E（https://codeforces.com/contest/1665/problem/E）segment_tree
1478E（https://codeforces.com/contest/1478/problem/E）RangeChangeRangeSumMinMax|backward_thinking|implemention
1679E（https://codeforces.com/contest/1679/problem/B）RangeChangeRangeSumMinMax|range_change|range_sum

====================================AtCoder=====================================
ABC332F（https://atcoder.jp/contests/abc332/tasks/abc332_f）RangeAddMulRangeSum

=====================================AcWing=====================================
3805（https://www.acwing.com/problem/content/3808/）RangeAddRangeMin

=====================================LibraryChecker=====================================
Range Affine Point Get（https://judge.yosupo.jp/problem/range_affine_point_get）RangeAddMulRangeSum
Range Affine Range Sum（https://judge.yosupo.jp/problem/range_affine_range_sum）RangeAddMulRangeSum


"""
import random
from collections import defaultdict
from math import inf
from typing import List

from sortedcontainers import SortedList

from src.basis.binary_search.template import BinarySearch
from src.data_structure.segment_tree.template import RangeAscendRangeMax, RangeDescendRangeMin, \
    RangeAddRangeSumMinMax, SegmentTreeRangeUpdateXORSum, SegmentTreeRangeUpdateChangeQueryMax, \
    SegmentTreeRangeXORQuery, SegmentTreePointChangeLongCon, SegmentTreeRangeSqrtSum, SegmentTreeRangeAndOrXOR, \
    RangeChangeRangeOr, \
    SegmentTreeRangeUpdateAvgDev, SegmentTreePointUpdateRangeMulQuery, \
    RangeChangeRangeSumMinMaxDynamic, SegmentTreeLongestSubSame, \
    RangeOrRangeAnd, RangeChangeRangeSumMinMax, RangeKSmallest, PointChangeRangeMaxNonEmpConSubSum, \
    RangeAscendRangeMaxBinarySearchFindLeft, RangeAddMulRangeSum
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_2213(s: str, word: str, indices: List[int]) -> List[int]:
        """
        url: https://leetcode.com/problems/longest-substring-of-one-repeating-character/
        tag: segment_tree|sub_consequence|range_query|range_merge
        """
        # 单点字母更新，最长具有相同字母的连续子数组查询
        n = len(s)
        tree = SegmentTreeLongestSubSame(n, [ord(w) - ord("a") for w in s])
        ans = []
        for i, w in zip(indices, word):
            ans.append(tree.update_point(i, i, 0, n - 1, ord(w) - ord("a"), 1))
        return ans

    @staticmethod
    def lib_check_1(ac=FastIO()):
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        mod = 998244353
        tree = RangeAddMulRangeSum(n, mod)
        tree.build(nums)
        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 0:
                ll, rr, b, c = lst[1:]
                tree.range_add_mul(ll, rr - 1, b, "mul")
                tree.range_add_mul(ll, rr - 1, c, "add")
            else:
                i = lst[1]
                ac.st(tree.range_sum(i, i))
        return

    @staticmethod
    def lib_check_2(ac=FastIO()):
        n, q = ac.read_list_ints()
        nums = ac.read_list_ints()
        mod = 998244353
        tree = RangeAddMulRangeSum(n, mod)
        tree.build(nums)
        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 0:
                ll, rr, b, c = lst[1:]
                tree.range_add_mul(ll, rr - 1, b, "mul")
                tree.range_add_mul(ll, rr - 1, c, "add")
            else:
                ll, rr = lst[1:]
                ac.st(tree.range_sum(ll, rr - 1))
        return

    @staticmethod
    def lc_2569_1(nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # 01segment_tree|区间翻转与求和，也可以BitSet
        n = len(nums1)
        tree = SegmentTreeRangeUpdateXORSum(n)
        tree.build(nums1)
        ans = []
        s = sum(nums2)
        for op, x, y in queries:
            if op == 1:
                tree.update_range(x, y, 0, n - 1, 1, 1)
            elif op == 2:
                s += tree.query_sum(0, n - 1, 0, n - 1, 1) * x
            else:
                ans.append(s)
        return ans

    @staticmethod
    def lg_p1904(ac=FastIO()):

        # segment_tree|，区间更新最大值并单点查询天际线
        high = 10 ** 4
        segment = RangeAscendRangeMax(high)
        segment.build([0] * high)
        nums = set()
        while True:
            s = ac.read_str()
            if not s:
                break
            x, h, y = [int(w) for w in s.split() if w]
            nums.add(x)
            nums.add(y)
            segment.range_ascend(x, y - 1, h)
        nums = sorted(list(nums))
        n = len(nums)
        height = [segment.range_max(num, num) for num in nums]
        ans = []
        pre = -1
        for i in range(n):
            if height[i] != pre:
                ans.extend([nums[i], height[i]])
                pre = height[i]
        ac.lst(ans)
        return

    @staticmethod
    def cf_1216f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1216/problem/F
        tag: segment_tree|dp|monotonic_queue
        """
        # segment_tree||DP
        n, k = ac.read_list_ints()
        s = ac.read_str()
        tree = RangeDescendRangeMin(n)
        for i in range(n):
            if s[i] == "1":
                left = ac.max(0, i - k)
                right = ac.min(n - 1, i + k)
                pre = tree.range_min(left - 1, i - 1) if left else 0
                cur = pre + i + 1
                tree.range_descend(i, right, cur)
            else:
                pre = tree.range_min(i - 1, i - 1) if i else 0
                cur = pre + i + 1
                tree.range_descend(i, i, cur)
        ac.st(tree.range_min(n - 1, n - 1))
        return

    @staticmethod
    def cf_1478e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1478/problem/E
        tag: RangeChangeRangeSumMinMax|backward_thinking|implemention
        """
        for _ in range(ac.read_int()):
            def check():
                n, q = ac.read_list_ints()
                s = [int(w) for w in ac.read_str()]
                t = [int(w) for w in ac.read_str()]
                queries = [ac.read_list_ints_minus_one() for _ in range(q)]
                queries.reverse()
                tree = RangeChangeRangeSumMinMax(n)
                tree.build(t)
                for ll, rr in queries:
                    cur_sum = tree.range_sum(ll, rr)
                    if cur_sum < rr - ll + 1 - cur_sum:
                        tree.range_change(ll, rr, 0)
                    elif cur_sum > rr - ll + 1 - cur_sum:
                        tree.range_change(ll, rr, 1)
                    else:
                        ac.st("NO")
                        return
                if tree.get() == s:
                    ac.st("YES")
                else:
                    ac.st("NO")
                return

            check()
        return

    @staticmethod
    def cf_1665e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1665/problem/E
        tag: segment_tree
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            tree = RangeKSmallest(n, 31)
            tree.build(nums)
            for _ in range(ac.read_int()):
                ll, rr = ac.read_list_ints_minus_one()
                lst = tree.range_k_smallest(ll, rr)
                ans = inf
                m = len(lst)
                for i in range(m):
                    x = lst[i]
                    if x > ans:
                        break
                    for j in range(i + 1, m):
                        y = lst[j]
                        if x > ans:
                            break
                        ans = ac.min(ans, x | y)
                ac.st(ans)
        return

    @staticmethod
    def lc_218(buildings: List[List[int]]) -> List[List[int]]:
        """
        url: https://leetcode.com/problems/the-skyline-problem/solution/by-liupengsay-isfo/
        tag: segment_tree|RangeChangeRangeMax
        """
        # segment_tree|discretization区间且持续增|最大值
        pos = set()
        for left, right, _ in buildings:
            pos.add(left)
            pos.add(right)
        lst = sorted(list(pos))
        n = len(lst)
        dct = {x: i for i, x in enumerate(lst)}
        # discretization更新segment_tree|
        segment = RangeAscendRangeMax(n)
        segment.build([0] * n)
        for left, right, height in buildings:
            segment.range_ascend(dct[left], dct[right] - 1, height)
        # 按照端点关键点查询
        pre = -1
        ans = []
        for pos in lst:
            h = segment.range_max(dct[pos], dct[pos])
            if h != pre:
                ans.append([pos, h])
                pre = h
        return ans

    @staticmethod
    def cf_380c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/380/C
        tag: segment_tree|range_merge|sub_consequence|bracket
        """
        word = []
        queries = []
        # segment_tree|divide_and_conquer并dp合并
        n = len(word)
        a = [0] * (4 * n)
        b = [0] * (4 * n)
        c = [0] * (4 * n)

        @ac.bootstrap
        def update(left, r, s, t, i):
            if s == t:
                if word[s - 1] == ")":
                    c[i] = 1
                else:
                    b[i] = 1
                a[i] = 0
                yield

            m = s + (t - s) // 2
            if left <= m:
                yield update(left, r, s, m, i << 1)
            if r > m:
                yield update(left, r, m + 1, t, i << 1 | 1)

            match = min(b[i << 1], c[i << 1 | 1])
            a[i] = a[i << 1] + a[i << 1 | 1] + 2 * match
            b[i] = b[i << 1] + b[i << 1 | 1] - match
            c[i] = c[i << 1] + c[i << 1 | 1] - match
            yield

        @ac.bootstrap
        def query(left, r, s, t, i):
            if left <= s and t <= r:
                d[i] = [a[i], b[i], c[i]]
                yield

            a1 = b1 = c1 = 0
            m = s + (t - s) // 2
            if left <= m:
                yield query(left, r, s, m, i << 1)
                a2, b2, c2 = d[i << 1]
                match = min(b1, c2)
                a1 += a2 + 2 * match
                b1 += b2 - match
                c1 += c2 - match
            if r > m:
                yield query(left, r, m + 1, t, i << 1 | 1)
                a2, b2, c2 = d[i << 1 | 1]
                match = min(b1, c2)
                a1 += a2 + 2 * match
                b1 += b2 - match
                c1 += c2 - match
            d[i] = [a1, b1, c1]
            yield

        update(1, n, 1, n, 1)
        ans = []
        for x, y in queries:
            d = defaultdict(list)
            query(x, y, 1, n, 1)
            ans.append(d[1][0])
        return ans

    @staticmethod
    def lg_p3372(ac=FastIO()):
        # segment_tree| 区间增减 与区间和查询
        n, m = ac.read_list_ints()
        segment = RangeAddRangeSumMinMax(n)
        segment.build(ac.read_list_ints())

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, k = lst[1:]
                segment.range_add(x - 1, y - 1, k)
            else:
                x, y = lst[1:]
                ac.st(segment.range_sum(x - 1, y - 1))
        return

    @staticmethod
    def lg_p3870(ac=FastIO()):
        # 区间异或 0 与 1 翻转
        n, m = ac.read_list_ints()
        segment = SegmentTreeRangeUpdateXORSum(n)

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 0:
                x, y = lst[1:]
                segment.update_range(x - 1, y - 1, 0, n - 1, 1, 1)
            else:
                x, y = lst[1:]
                ac.st(segment.query_sum(x - 1, y - 1, 0, n - 1, 1))
        return

    @staticmethod
    def lg_p1438(ac=FastIO()):
        # diff_array|区间增减|segment_tree|查询区间和
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        segment = RangeAddRangeSumMinMax(n)

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, k, d = lst[1:]
                if x == y:
                    segment.range_add(x - 1, x - 1, k)
                    if y <= n - 1:
                        segment.range_add(y, y, -k)
                else:
                    # diff_array|区间的等差数列|减
                    segment.range_add(x - 1, x - 1, k)
                    segment.range_add(x, y - 1, d)
                    cnt = y - x
                    if y <= n - 1:
                        segment.range_add(y, y, -cnt * d - k)
            else:
                x = lst[1]
                ac.st(segment.range_sum(0, x - 1) + nums[x - 1])
        return

    @staticmethod
    def lg_p1253(ac=FastIO()):

        # 区间增减与区间修改并segment_tree|查询区间和
        n, m = ac.read_list_ints()
        segment = SegmentTreeRangeUpdateChangeQueryMax(ac.read_list_ints())

        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, k = lst[1:]
                segment.update(x - 1, y - 1, 0, n - 1, k, 1, 1)
            elif lst[0] == 2:
                x, y, k = lst[1:]
                segment.update(x - 1, y - 1, 0, n - 1, k, 2, 1)
            else:
                x, y = lst[1:]
                ac.st(segment.query_max(x - 1, y - 1, 0, n - 1, 1))
        return

    @staticmethod
    def lg_p3373(ac=FastIO()):
        n, q, mod = ac.read_list_ints()
        tree = RangeAddMulRangeSum(n, mod)
        nums = ac.read_list_ints()
        tree.build(nums)
        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                x, y, k = lst[1:]
                tree.range_add_mul(x - 1, y - 1, k, "mul")
            elif lst[0] == 2:
                x, y, k = lst[1:]
                tree.range_add_mul(x - 1, y - 1, k, "add")
            else:
                x, y = lst[1:]
                ans = tree.range_sum(x - 1, y - 1)
                ac.st(ans)
        return

    @staticmethod
    def lg_p4513(ac=FastIO()):
        n, m = ac.read_list_ints()
        segment = PointChangeRangeMaxNonEmpConSubSum(n, 1000)
        segment.build([ac.read_int() for _ in range(n)])
        for _ in range(m):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                a, b = lst[1:]
                a, b = ac.min(a, b), ac.max(a, b)
                ans = segment.range_max_non_emp_con_sub_sum(a - 1, b - 1)[0]
                ac.st(ans)
            else:
                a, s = lst[1:]
                segment.range_change(a - 1, a - 1, s)
        return

    @staticmethod
    def lg_p1471(ac=FastIO()):
        # 区间增减，维护区间和与区间数字平方的和，以均差与方差
        n, m = ac.read_list_ints()
        tree = SegmentTreeRangeUpdateAvgDev(n)
        tree.build(ac.read_list_floats())
        for _ in range(m):
            lst = ac.read_list_floats()
            if lst[0] == 1:
                x, y, k = lst[1:]
                x = int(x)
                y = int(y)
                tree.update(x - 1, y - 1, 0, n - 1, k, 1)
            elif lst[0] == 2:
                x, y = lst[1:]
                x = int(x)
                y = int(y)
                ans = (tree.query(x - 1, y - 1, 0, n - 1, 1)[0]) / (y - x + 1)
                ac.st("%.4f" % ans)
            else:
                x, y = lst[1:]
                x = int(x)
                y = int(y)
                avg, avg_2 = tree.query(x - 1, y - 1, 0, n - 1, 1)
                ans = -(avg * avg / (y - x + 1)) ** 2 + avg_2 / (y - x + 1)
                ac.st("%.4f" % ans)
        return

    @staticmethod
    def lg_p6627(ac=FastIO()):
        # segment_tree|维护和查询区间异或值
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nodes = {0, -10 ** 9 - 1, 10 ** 9 + 1}
        for lst in nums:
            for va in lst[1:-1]:
                nodes.add(va)
                nodes.add(va - 1)
                nodes.add(va + 1)
        nodes = sorted(list(nodes))
        n = len(nodes)
        ind = {num: i for i, num in enumerate(nodes)}
        tree = SegmentTreeRangeXORQuery(n)
        arr = [0] * n
        for lst in nums:
            if lst[0] == 1:
                a, b, w = lst[1:]
                if a > b:
                    a, b = b, a
                tree.update_range(ind[a], ind[b], 0, n - 1, w, 1)
            elif lst[0] == 2:
                a, w = lst[1:]
                arr[ind[a]] ^= w  # 数组代替
                # tree.update_point(ind[a], ind[a], 0, n-1, w, 1)
            else:
                a, w = lst[1:]
                tree.update_range(0, n - 1, 0, n - 1, w, 1)
                arr[ind[a]] ^= w  # 数组代替
                # tree.update_point(ind[a], ind[a], 0, n - 1, w, 1)
        ans = inf
        res = -inf
        for i in range(n):
            val = tree.query_point(i, i, 0, n - 1, 1) ^ arr[i]
            if val > res or (
                    val == res and (abs(ans) > abs(nodes[i]) or (abs(ans) == abs(nodes[i]) and nodes[i] > ans))):
                res = val
                ans = nodes[i]
        ac.lst([res, ans])
        return

    @staticmethod
    def lg_p6492(ac=FastIO()):
        # 单点修改，查找最长的01交替字符子串连续区间
        n, q = ac.read_list_ints()
        tree = SegmentTreePointChangeLongCon(n)
        for _ in range(q):
            i = ac.read_int() - 1
            tree.update(i, i, 0, n - 1, 1)
            ac.st(tree.query())
        return

    @staticmethod
    def lg_p4145(ac=FastIO()):
        # 区间值开方向下取整，区间和查询
        n = ac.read_int()
        tree = SegmentTreeRangeSqrtSum(n)
        tree.build(ac.read_list_ints())
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            a, b = [int(w) - 1 for w in lst[1:]]
            if a > b:
                a, b = b, a
            if lst[0] == 0:
                tree.change(a, b, 0, n - 1, 1)
            else:
                ac.st(tree.query_sum(a, b, 0, n - 1, 1))
        return

    @staticmethod
    def lc_2940(heights: List[int], queries: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.com/problems/find-building-where-alice-and-bob-can-meet/
        tag: segment_tree_binary_search
        """
        n = len(heights)
        tree = RangeAscendRangeMaxBinarySearchFindLeft(n)
        tree.build(heights)
        ans = []
        for ll, rr in queries:
            if ll > rr:
                ll, rr = rr, ll
            if heights[ll] < heights[rr]:
                ans.append(rr)
                continue
            if ll == rr:
                ans.append(ll)
                continue
            if rr == n - 1:
                ans.append(-1)
                continue
            h = heights[ll] if heights[ll] > heights[rr] else heights[rr]
            left = tree.binary_search_find_left(rr + 1, n - 1, h + 1)

            ans.append(left)
        return ans

    @staticmethod
    def lg_2572(ac=FastIO()):
        # 区间修改成01或者翻转，区间查询最多有多少连续的1，以及总共有多少1
        def check(tmp):
            ans = pre = 0
            for num in tmp:
                if num:
                    pre += 1
                else:
                    ans = ans if ans > pre else pre
                    pre = 0
            ans = ans if ans > pre else pre
            return ans

        # n, m = ac.read_list_ints()
        for s in range(100):
            # random.seed(s)
            print(f"seed_{s}")
            n = random.randint(1, 1000)
            m = random.randint(1, 1000)
            tree = SegmentTreeRangeAndOrXOR(n)
            # tree.build(ac.read_list_ints())
            nums = [random.randint(1, 1) for _ in range(n)]
            tree.build(nums)
            for _ in range(m):
                lst = [random.randint(0, 4), random.randint(0, n - 1), random.randint(0, n - 1)]
                # lst = ac.read_list_ints()
                left, right = lst[1:]
                if left > right:
                    left, right = right, left
                if lst[0] <= 2:
                    tree.update(left, right, 0, n - 1, lst[0], 1)
                    if lst[0] == 0:
                        for i in range(left, right + 1):
                            nums[i] = 0
                    elif lst[0] == 1:
                        for i in range(left, right + 1):
                            nums[i] = 1
                    else:
                        for i in range(left, right + 1):
                            nums[i] = 1 - nums[i]
                    assert nums == [tree.query_sum(i, i, 0, n - 1, 1) for i in range(n)]
                elif lst[0] == 3:
                    # print("\n")
                    # print(nums)
                    # print([tree.query_sum(i, i, 0, n-1, 1) for i in range(n)])
                    assert sum(nums[left:right + 1]) == tree.query_sum(left, right, 0, n - 1, 1)
                    # ac.st(tree.query_sum(left, right, 0, n-1, 1))
                else:
                    # print("\n")
                    # print(nums)
                    # print([tree.query_sum(i, i, 0, n-1, 1) for i in range(n)])
                    # print(tree.query_max_length(left, right, 0, n - 1, 1)[0], check(nums[left:right+1]))
                    assert tree.query_max_length(left, right, 0, n - 1, 1)[0] == check(nums[left:right + 1])
                    # ac.st(tree.query_max_length(left, right, 0, n - 1, 1)[0])
        return

    @staticmethod
    def lg_p1558(ac=FastIO()):
        # segment_tree|区间修改，区间或查询
        n, t, q = ac.read_list_ints()
        tree = RangeChangeRangeOr(n)
        tree.range_change(0, n - 1, 1)
        for _ in range(q):
            lst = ac.read_list_strs()
            if lst[0] == "C":
                a, b, c = [int(w) for w in lst[1:]]
                if a > b:
                    a, b = b, a
                # 修改区间值
                tree.range_change(a - 1, b - 1, 1 << (c - 1))
            else:
                a, b = [int(w) for w in lst[1:]]
                if a > b:
                    a, b = b, a
                # 区间值或查询
                ac.st(bin(tree.range_or(a - 1, b - 1)).count("1"))
        return

    @staticmethod
    def lg_p3740(ac=FastIO()):
        # discretizationsegment_tree|区间修改与单点查询
        n, m = ac.read_list_ints()
        nums = []
        while len(nums) < m * 2:
            nums.extend(ac.read_list_ints())
        nums = [nums[2 * i:2 * i + 2] for i in range(m)]
        nodes = set()
        nodes.add(1)
        nodes.add(n)
        for a, b in nums:
            nodes.add(a)
            nodes.add(b)
            # discretization特别注意需要增|右端点连续区间的区分
            nodes.add(b + 1)
        nodes = list(sorted(nodes))
        ind = {num: i for i, num in enumerate(nodes)}

        # 区间修改
        n = len(nodes)
        tree = RangeChangeRangeSumMinMax(n)
        tree.build([0] * n)
        for i in range(m):
            a, b = nums[i]
            tree.range_change(ind[a], ind[b], i + 1)

        # 单点查询
        ans = tree.get()
        ac.st(len(set(c for c in ans if c)))
        return

    @staticmethod
    def lg_p4588(ac=FastIO()):
        # 转化为segment_tree|单点值修改与区间乘积mod|
        for _ in range(ac.read_int()):
            q, mod = ac.read_list_ints()
            tree = SegmentTreePointUpdateRangeMulQuery(q, mod)
            for i in range(q):
                op, num = ac.read_list_ints()
                if op == 1:
                    tree.update(i, i, 0, q - 1, num % mod, 1)
                else:
                    tree.update(num - 1, num - 1, 0, q - 1, 1, 1)
                ac.st(tree.cover[1])
        return

    @staticmethod
    def lg_p8081(ac=FastIO()):
        # segment_tree|区间修改、区间|和查询
        n = ac.read_int()
        nums = ac.read_list_ints()
        tree = RangeChangeRangeSumMinMax(n)
        pre = 0
        ceil = 0
        for i in range(n):
            if nums[i] < 0:
                pre += 1
            else:
                if pre:
                    ceil = max(ceil, pre)
                    low, high = i - 3 * pre, i - pre - 1
                    if high >= 0:
                        tree.range_change(ac.max(0, low), high, 1)
                pre = 0
        if pre:
            ceil = max(ceil, pre)
            low, high = n - 3 * pre, n - pre - 1
            if high >= 0:
                tree.range_change(ac.max(0, low), high, 1)

        ans = tree.range_sum(0, n - 1)
        pre = 0
        res = 0
        for i in range(n):
            if nums[i] < 0:
                pre += 1
            else:
                if pre == ceil:
                    low, high = i - 4 * pre, i - 3 * pre - 1
                    low = ac.max(low, 0)
                    if low <= high:
                        res = ac.max(res, high - low + 1 - tree.range_sum(low, high))
                pre = 0
        if pre == ceil:
            low, high = n - 4 * pre, n - 3 * pre - 1
            low = ac.max(low, 0)
            if low <= high:
                res = ac.max(res, high - low + 1 - tree.range_sum(low, high))
        ac.st(ans + res)
        return

    @staticmethod
    def lg_p8812(ac=FastIO()):
        # segment_tree|查询和更新区间最小值
        n, m = ac.read_list_ints()
        goods = [[] for _ in range(n)]
        for _ in range(m):
            s, t, p, c = ac.read_list_ints()
            for _ in range(c):
                a, b = ac.read_list_ints()
                a -= 1
                goods[a].append([1, 10 ** 9 + 1, b])
                b = b * p // 100
                goods[a].append([s, t, b])

        for i in range(n):
            nodes = {0, 10 ** 9 + 1}
            for s, t, _ in goods[i]:
                nodes.add(s - 1)
                nodes.add(s)
                nodes.add(t)
                nodes.add(t + 1)
            nodes = sorted(list(nodes))
            ind = {node: i for i, node in enumerate(nodes)}
            k = len(ind)
            tree = RangeDescendRangeMin(k)
            for s, t, b in goods[i]:
                tree.range_descend(ind[s], ind[t], b)
            res = []
            for x in range(k):
                val = tree.range_min(x, x)
                if val == inf:
                    continue
                if not res or res[-1][2] != val:
                    res.append([nodes[x], nodes[x], val])
                else:
                    res[-1][1] = nodes[x]

            goods[i] = [r[:] for r in res]

        nodes = {0, 10 ** 9 + 1}
        for i in range(n):
            for s, t, _ in goods[i]:
                nodes.add(s)
                nodes.add(t)
        nodes = sorted(list(nodes))
        ind = {node: i for i, node in enumerate(nodes)}
        k = len(ind)
        diff = [0] * k
        for i in range(n):
            for s, t, b in goods[i]:
                diff[ind[s]] += b
                if ind[t] + 1 < k:
                    diff[ind[t] + 1] -= b
        diff = ac.accumulate(diff)[2:]
        ac.st(min(diff))
        return

    @staticmethod
    def cf_482b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/482/B
        tag: segment_tree|RangeOrRangeAnd
        """
        n, m = ac.read_list_ints()
        tree = RangeOrRangeAnd(n)
        nums = [ac.read_list_ints() for _ in range(m)]
        for a, b, c in nums:
            if c:
                tree.range_or(a, b, c)
        if all(tree.range_and(a, b) == c for a, b, c in nums):
            ac.st("YES")
            ac.lst(tree.get())
        else:
            ac.st("NO")
        return

    @staticmethod
    def cf_987c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/987/C
        tag: brute_force|segment_tree|prefix_suffix
        """
        # brute_force中间数组，segment_tree|维护prefix_suffix最小值
        n = ac.read_int()
        s = ac.read_list_ints()
        c = ac.read_list_ints()
        ind = {num: i for i, num in enumerate(sorted(list(set(s + c + [0] + [10 ** 9 + 1]))))}
        m = len(ind)
        post = [inf] * n
        tree = RangeDescendRangeMin(m)
        for i in range(n - 1, -1, -1):
            tree.range_descend(ind[s[i]], ind[s[i]], c[i])
            post[i] = tree.range_min(ind[s[i]] + 1, m - 1)

        ans = inf
        tree = RangeDescendRangeMin(m)
        for i in range(n):
            if 1 <= i <= n - 2:
                cur = c[i] + tree.range_min(0, ind[s[i]] - 1) + post[i]
                ans = ac.min(ans, cur)
            tree.range_descend(ind[s[i]], ind[s[i]], c[i])
        ac.st(ans if ans < inf else -1)
        return

    @staticmethod
    def lc_1851(intervals: List[List[int]], queries: List[int]) -> List[int]:
        """
        url: https://leetcode.com/problems/minimum-interval-to-include-each-query/
        tag: segment_tree|RangeChangeRangeMin|offline_query|priority_queue
        """
        # 区间更新最小值、单点查询
        port = []
        for inter in intervals:
            port.extend(inter)
        port.extend(queries)
        lst = sorted(list(set(port)))

        ind = {num: i for i, num in enumerate(lst)}
        ceil = len(lst)
        tree = RangeDescendRangeMin(ceil)
        for a, b in intervals:
            tree.range_descend(ind[a], ind[b], b - a + 1)
        ans = [tree.range_min(ind[num], ind[num]) for num in queries]
        return [x if x != inf else -1 for x in ans]

    @staticmethod
    def lc_1340(nums: List[int], d: int) -> int:
        """
        url: https://leetcode.com/problems/jump-game-v/
        tag: segment_tree|linear_dp
        """

        # 可以segment_tree|DP解决
        n = len(nums)
        post = [n - 1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                post[stack.pop()] = i - 1
            stack.append(i)

        pre = [0] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                pre[stack.pop()] = i + 1
            stack.append(i)

        # 分bucket_sort转移
        dct = defaultdict(list)
        for i, num in enumerate(nums):
            dct[num].append(i)
        tree = RangeAscendRangeMax(n)
        tree.build([0] * n)
        for num in sorted(dct):
            cur = []
            for i in dct[num]:
                left, right = pre[i], post[i]
                if left < i - d:
                    left = i - d
                if right > i + d:
                    right = i + d
                x = tree.range_max(left, right)
                cur.append([x + 1, i])

            for x, i in cur:
                tree.range_ascend(i, i, x)
        return tree.range_max(0, n - 1)

    @staticmethod
    def abc_332f(ac=FastIO()):
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        mod = 998244353
        tree = RangeAddMulRangeSum(n, mod)
        tree.build(nums)
        for _ in range(m):
            ll, rr, xx = ac.read_list_ints()
            length = rr - ll + 1
            pre = ((length - 1) * pow(length, -1, mod)) % mod
            tree.range_add_mul(ll - 1, rr - 1, pre, "mul")
            val = (xx * pow(length, -1, mod)) % mod
            tree.range_add_mul(ll - 1, rr - 1, val, "add")
        ac.lst(tree.get())
        return

    @staticmethod
    def ac_3805(ac=FastIO()):
        # 区间增减与最小值查询
        n = ac.read_int()
        tree = RangeAddRangeSumMinMax(n)
        tree.build(ac.read_list_ints())
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if len(lst) == 2:
                ll, r = lst
                if ll <= r:
                    ac.st(tree.range_min(ll, r))
                else:
                    ans1 = tree.range_min(ll, n - 1)
                    ans2 = tree.range_min(0, r)
                    ac.st(ac.min(ans1, ans2))
            else:
                ll, r, d = lst
                if ll <= r:
                    tree.range_add(ll, r, d)
                else:
                    tree.range_add(ll, n - 1, d)
                    tree.range_add(0, r, d)
        return

    @staticmethod
    def ac_5037_1(ac=FastIO()):
        # 同CF242E，二十多个01segment_tree|维护区间异或与区间|和
        n = ac.read_int()
        nums = ac.read_list_ints()
        tree = [SegmentTreeRangeUpdateXORSum(n) for _ in range(22)]
        for j in range(22):
            lst = [1 if nums[i] & (1 << j) else 0 for i in range(n)]
            tree[j].build(lst)
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                ll, rr = lst[1:]
                ll -= 1
                rr -= 1
                ans = sum((1 << j) * tree[j].query_sum(ll, rr, 0, n - 1, 1) for j in range(22))
                ac.st(ans)
            else:
                ll, rr, xx = lst[1:]
                ll -= 1
                rr -= 1
                for j in range(22):
                    if (1 << j) & xx:
                        tree[j].update(ll, rr, 0, n - 1, 1, 1)

        return


class CountIntervalsLC2276:

    def __init__(self):
        # dynamic_segment_tree
        self.n = 10 ** 9 + 7
        self.segment_tree = RangeChangeRangeSumMinMaxDynamic(self.n)

    def add(self, left: int, right: int) -> None:
        self.segment_tree.range_change(left, right, 1)

    def count(self) -> int:
        return self.segment_tree.cover[1]


class BookMyShowLC2286:

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.tree = RangeAddRangeSumMinMax(n)
        self.cnt = [0] * n
        self.null = SortedList(list(range(n)))

    def gather(self, k: int, max_row: int) -> List[int]:
        max_row += 1
        low = self.tree.range_min(0, max_row - 1)
        if self.m - low < k:
            return []

        def check(x):
            return self.m - self.tree.range_min(0, x) >= k

        # binary_search|segment_tree|维护最小值与和
        y = BinarySearch().find_int_left(0, max_row - 1, check)
        self.cnt[y] += k
        self.tree.range_add(y, y, k)
        if self.cnt[y] == self.m:
            self.null.discard(y)
        return [y, self.cnt[y] - k]

    def scatter(self, k: int, max_row: int) -> bool:
        max_row += 1
        s = self.tree.range_sum(0, max_row - 1)
        if self.m * max_row - s < k:
            return False
        while k:
            x = self.null[0]
            rest = k if k < self.m - self.cnt[x] else self.m - self.cnt[x]
            k -= rest
            self.cnt[x] += rest
            self.tree.range_add(x, x, rest)
            if self.cnt[x] == self.m:
                self.null.pop(0)
        return True