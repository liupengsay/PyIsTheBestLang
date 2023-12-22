"""
Algorithm：range_merge_to_disjoint|minimum_range_cover|counter|maximum_disjoint_range
Description：sometimes cooperation with diff_array or tree_array or segment_tree
minimum_point_cover_range|minimum_group_range_disjoint|maximum_point_cover_range|bipartite_graph

====================================LeetCode====================================
45（https://leetcode.cn/problems/jump-game-ii/）minimum_range_cover
452（https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/）greedy|maximum_disjoint_range
1326（https://leetcode.cn/problems/minimum-number-of-taps-to-open-to-water-a-garden/）range_merge_to_disjoint
1024（https://leetcode.cn/problems/video-stitching/）minimum_range_cover
1520（https://leetcode.cn/problems/maximum-number-of-non-overlapping-substrings/）maximum_disjoint_range
1353（https://leetcode.cn/problems/maximum-number-of-events-that-can-be-attended/）greedy|minimum_point_cover_range
2406（https://leetcode.cn/problems/divide-intervals-into-minimum-number-of-groups/）minimum_group_range_disjoint|greedy|diff_array|counter
435（https://leetcode.cn/problems/non-overlapping-intervals/）maximum_disjoint_range|greedy|binary_search|dp
763（https://leetcode.cn/problems/partition-labels/）range_merge_to_disjoint
6313（https://leetcode.cn/contest/biweekly-contest-99/problems/count-ways-to-group-overlapping-ranges/）range_merge_to_disjoint|fast_power|counter
2345（https://leetcode.cn/problems/finding-the-number-of-visible-mountains/）partial_order|range_include
757（https://leetcode.cn/problems/set-intersection-size-at-least-two/）greedy|minimum_point_cover_range
2589（https://leetcode.cn/problems/minimum-time-to-complete-all-tasks/）greedy|minimum_point_cover_range
32（https://leetcode.cn/problems/t3fKg1/）greedy|minimum_point_cover_range

=====================================LuoGu======================================
P2082（https://www.luogu.com.cn/problem/P2082）range_merge_to_disjoint
P2434（https://www.luogu.com.cn/problem/P2434）range_merge_to_disjoint
P2970（https://www.luogu.com.cn/problem/P2970）maximum_disjoint_range|greedy|binary_search|dp
P6123（https://www.luogu.com.cn/problem/P6123）range_merge_to_disjoint
P2684（https://www.luogu.com.cn/problem/P2684）minimum_range_cover
P1233（https://www.luogu.com.cn/problem/P1233）partial_order|lis|range_include
P1496（https://www.luogu.com.cn/problem/P1496）range_merge_to_disjoint
P1668（https://www.luogu.com.cn/problem/P1668）minimum_range_cover
P2887（https://www.luogu.com.cn/problem/P2887）maximum_point_cover_range
P3661（https://www.luogu.com.cn/problem/P3661）greedy
P3737（https://www.luogu.com.cn/problem/P3737）greedy|range_cover
P5199（https://www.luogu.com.cn/problem/P5199）greedy|range_include
P1868（https://www.luogu.com.cn/problem/P1868）liner_dp|binary_search|maximum_disjoint_range
P2439（https://www.luogu.com.cn/problem/P2439）liner_dp|binary_search|maximum_disjoint_range
P1325（https://www.luogu.com.cn/problem/P1325）sort|greedy|minimum_range_cover

===================================CodeForces===================================
827A（https://codeforces.com/problemset/problem/827/A）range_merge_to_disjoint|greedy
652D（https://codeforces.com/problemset/problem/652/D）partial_order|range_include
1426D（https://codeforces.com/problemset/problem/1426/D）greedy|minimum_point_cover_range
1102E（https://codeforces.com/contest/1102/problem/E）range_merge_to_disjoint
1141F2（https://codeforces.com/contest/1141/problem/F2）prefix_sum|brute_force|maximum_disjoint_range

=====================================AcWing=====================================
112（https://www.acwing.com/problem/content/114/）greedy
4421（https://www.acwing.com/problem/content/4424/）minimum_range_cover


"""
import bisect
from collections import defaultdict, deque
from src.utils.fast_io import inf
from typing import List

from src.basis.range.template import Range
from src.data_structure.sorted_list.template import LocalSortedList
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lg_p1496(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1496
        tag: range_merge_to_disjoint
        """

        n = ac.read_int()
        lst = []
        for _ in range(n):
            a, b = [int(w) for w in input().strip().split() if w]
            lst.append([a, b])
        ans = sum(b - a for a, b in Range().range_merge_to_disjoint(lst))
        ac.st(ans)
        return

    @staticmethod
    def lc_45(nums):
        """
        url: https://leetcode.cn/problems/jump-game-ii/
        tag: minimum_range_cover
        """
        n = len(nums)
        lst = [[i, min(n - 1, i + nums[i])] for i in range(n)]
        if n == 1:
            return 0
        return Range().minimum_range_cover(0, n - 1, lst)

    @staticmethod
    def lc_1326_1(n, ranges):
        """
        url: https://leetcode.cn/problems/minimum-number-of-taps-to-open-to-water-a-garden/
        tag: range_merge_to_disjoint
        """
        m = n + 1
        lst = []
        for i in range(m):
            lst.append([max(i - ranges[i], 0), i + ranges[i]])
        return Range().minimum_range_cover(0, n, lst, True)

    @staticmethod
    def lc_1326_2(n: int, ranges: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-taps-to-open-to-water-a-garden/
        tag: range_merge_to_disjoint
        """

        lst = []
        for i, r in enumerate(ranges):
            a, b = i - r, i + r
            a = a if a > 0 else 0
            b = b if b < n else n
            lst.append([a, b])
        return Range().minimum_interval_coverage(lst, n, True)

    @staticmethod
    def lc_1024_1(clips, time) -> int:
        """
        url: https://leetcode.cn/problems/video-stitching/
        tag: minimum_range_cover
        """

        return Range().minimum_range_cover(0, time, clips)

    @staticmethod
    def lc_1024_2(clips: List[List[int]], time: int) -> int:
        """
        url: https://leetcode.cn/problems/video-stitching/
        tag: minimum_range_cover
        """

        return Range().minimum_interval_coverage(clips, time, True)

    @staticmethod
    def lg_p2684(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2684
        tag: minimum_range_cover
        """
        n, t = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        ac.st(Range().minimum_range_cover(1, t, nums))
        return

    @staticmethod
    def lc_435(intervals):
        """
        url: https://leetcode.cn/problems/non-overlapping-intervals/
        tag: maximum_disjoint_range|greedy|binary_search|dp|classical
        """

        n = len(intervals)
        return n - Range().maximum_disjoint_range(intervals)

    @staticmethod
    def lc_763(s: str) -> List[int]:
        """
        url: https://leetcode.cn/problems/partition-labels/
        tag: range_merge_to_disjoint
        """

        dct = defaultdict(list)
        for i, w in enumerate(s):
            if len(dct[w]) >= 2:
                dct[w].pop()
            dct[w].append(i)
        lst = []
        for w in dct:
            lst.append([dct[w][0], dct[w][-1]])
        ans = Range().range_merge_to_disjoint(lst)
        return [y - x + 1 for x, y in ans]

    @staticmethod
    def lc_6313(ranges: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/count-ways-to-group-overlapping-ranges/
        tag: range_merge_to_disjoint|fast_power|counter
        """

        cnt = len(Range().range_merge_to_disjoint(ranges))
        mod = 10 ** 9 + 7
        return pow(2, cnt, mod)

    @staticmethod
    def cf_1102e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1102/problem/E
        tag: range_merge_to_disjoint
        """

        mod = 998244353
        n = ac.read_int()
        nums = ac.read_list_ints()
        dct = defaultdict(list)
        for i in range(n):
            dct[nums[i]].append(i)
        lst = [[dct[num][0], dct[num][-1]] for num in dct]
        ans = len(Range().range_merge_to_disjoint(lst))
        ac.st(pow(2, ans - 1, mod))
        return

    @staticmethod
    def cf_1426d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1426/D
        tag: greedy|minimum_point_cover_range|prefix_sum_hash
        """
        ac.read_int()
        nums = ac.read_list_ints()
        ans = pre = 0
        dct = {pre}
        for num in nums:
            if pre + num in dct:
                ans += 1
                pre = num
                dct = {0, pre}
            else:
                pre += num
                dct.add(pre)
        ac.st(ans)
        return

    @staticmethod
    def ac_112(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/114/
        tag: greedy|range|rever_thinking|minimum_point_cover_range
        """

        n, d = ac.read_list_ints()
        lst = [ac.read_list_ints() for _ in range(n)]
        if any(abs(y) > d for _, y in lst):
            ac.st(-1)
            return
        for i in range(n):
            x, y = lst[i]
            r = (d * d - y * y) ** 0.5
            lst[i] = [x - r, x + r]
        ac.st(Range().minimum_point_cover_range(lst))  # minimum_point_cover_range_2 also accepted
        return

    @staticmethod
    def lg_p1668(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1668
        tag: minimum_range_cover|classical
        """

        n, t = ac.read_list_ints()
        lst = [ac.read_list_ints() for _ in range(n)]
        ans = Range().minimum_range_cover(1, t, lst, False)
        ac.st(ans)
        return

    @staticmethod
    def lg_p1668_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1668
        tag: minimum_range_cover
        """

        n, t = ac.read_list_ints()
        t -= 1
        lst = [ac.read_list_ints_minus_one() for _ in range(n)]
        ans = Range().minimum_range_cover(0, t, lst, False)
        ac.st(ans)
        return

    @staticmethod
    def lg_p2887(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2887
        tag: maximum_point_cover_range|greedy|sorted_list|classical
        """

        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: it[1])
        pos = [ac.read_list_ints() for _ in range(m)]
        lst = LocalSortedList([(x, c) for x, c in pos])
        ans = 0
        for floor, ceil in nums:
            i = lst.bisect_left((floor, 0))
            if 0 <= i < len(lst) and floor <= lst[i][0] <= ceil:
                x, c = lst.pop(i)
                ans += 1
                if c - 1:
                    lst.add((x, c - 1))
        ac.st(ans)
        return

    @staticmethod
    def lg_p3661(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3661
        tag: greedy|maximum_point_cover_range|classical
        """

        n, m = ac.read_list_ints()
        lst = LocalSortedList([ac.read_int() for _ in range(n)])
        nums = [ac.read_list_ints() for _ in range(m)]
        nums.sort(key=lambda it: it[1])
        ans = 0
        for a, b in nums:
            i = lst.bisect_left(a)
            if 0 <= i < len(lst) and a <= lst[i] <= b:
                ans += 1
                lst.pop(i)
        ac.st(ans)
        return

    @staticmethod
    def lg_p3737(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3737
        tag: greedy|range|rever_thinking|minimum_point_cover_range
        """

        n, r = ac.read_list_ints()
        lst = []
        for i in range(n):
            x, y = ac.read_list_ints()
            if r * r < y * y:
                ac.st(-1)
                return
            d = (r * r - y * y) ** 0.5
            lst.append([x - d, x + d])
        ac.st(Range().minimum_point_cover_range(lst))
        return

    @staticmethod
    def lg_p5199(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5199
        tag: greedy|range_include|classical|partial_order
        """

        n = ac.read_int()
        nums = []
        for _ in range(n):
            x, y = ac.read_list_ints()
            nums.append([x - y, x + y])
        nums.sort(key=lambda it: [it[0], -it[1]])
        ans = 0
        pre = -inf
        for a, b in nums:
            if b > pre:
                ans += 1
                pre = b
        ac.st(ans)
        return

    @staticmethod
    def lc_1520(s: str) -> List[str]:
        """
        url: https://leetcode.cn/problems/maximum-number-of-non-overlapping-substrings/
        tag: maximum_disjoint_range
        """

        ind = defaultdict(deque)
        for i, w in enumerate(s):
            ind[w].append(i)

        lst = []
        for w in ind:
            x, y = ind[w][0], ind[w][-1]
            while True:
                x_pre, y_pre = x, y
                for v in ind:
                    i = bisect.bisect_right(ind[v], x)
                    j = bisect.bisect_left(ind[v], y) - 1
                    if 0 <= i <= j < len(ind[v]):
                        if ind[v][0] < x:
                            x = ind[v][0]
                        if ind[v][-1] > y:
                            y = ind[v][-1]
                if x == x_pre and y == y_pre:
                    break
            ind[w].appendleft(x)
            ind[w].append(y)
            lst.append([x, y])

        lst.sort(key=lambda ls: ls[1])
        ans = []
        for x, y in lst:
            if not ans or x > ans[-1][1]:
                ans.append([x, y])
        return [s[i: j + 1] for i, j in ans]

    @staticmethod
    def ac_4421(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4424/
        tag: minimum_range_cover
        """

        n, r = ac.read_list_ints()
        nums = ac.read_list_ints()
        lst = []
        for i in range(n):
            if nums[i]:
                a, b = i - r + 1, i + r - 1
                a = ac.max(a, 0)
                lst.append([a, b])
        ans = Range().minimum_range_cover(0, n - 1, lst, False)
        ac.st(ans)
        return
