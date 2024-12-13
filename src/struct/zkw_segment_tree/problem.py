"""
Algorithm：segment_tree|bisect_left
Description：range_sum|range_min|range_add|range_change|range_max|dynamic_segment_tree|defaulted_dict

====================================LeetCode====================================

=====================================LuoGu======================================

===================================CodeForces===================================
1093G（https://codeforces.com/contest/1093/problem/G）manhattan_distance|point_set|range_max_min_gap|classical

====================================AtCoder=====================================
ABC186F（https://atcoder.jp/contests/abc186/tasks/abc186_f）PointSetAddRangeSum|implemention|brain_teaser|brute_force|contribution_method
ABC179F（https://atcoder.jp/contests/abc179/tasks/abc179_f）zkw_segment_tree|implemention|brain_teaser
ABC178E（https://atcoder.jp/contests/abc178/tasks/abc178_e）PointUpdateRangeQuery|manhattan_distance|classical
ABC379D（https://atcoder.jp/contests/abc379/tasks/abc379_d）RangeAddPointGet|diff_array|classical
ABC382F（https://atcoder.jp/contests/abc382/tasks/abc382_f）RangeDescendRangeMin|implemention

=====================================AcWing=====================================

=====================================LibraryChecker=====================================


"""
import math
from collections import defaultdict
from typing import List

from src.struct.segment_tree.template import PointSetRangeMaxMinGap
from src.struct.zkw_segment_tree.template import PointSetAddRangeSum, RangeUpdatePointQuery, RangeAddPointGet, \
    PointUpdateRangeQuery, RangeDescendRangeMin, PointSetRangeMinCount
from src.util.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1093g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1093/problem/G
        tag: point_set|range_max_min_gap|classical
        """
        n, k = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        pre = [[1]]
        for _ in range(k - 1):
            cur = []
            for p in pre:
                cur.append(p + [1])
                cur.append(p + [-1])
            pre = cur[:]
        m = len(pre)
        assert m == 1 << (k - 1)
        tree = [PointSetRangeMaxMinGap(n) for _ in range(m)]
        for i in range(m):
            ls = pre[i]
            val = [sum(ls[j] * num[j] for j in range(k)) for num in nums]
            tree[i].build(val)
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                i = lst[1] - 1
                nums[i] = lst[2:]
                for x in range(m):
                    val = sum(pre[x][j] * nums[i][j] for j in range(k))
                    tree[x].point_set(i, val)
            else:
                ans = -math.inf
                ll, rr = [w - 1 for w in lst[1:]]
                for i in range(m):
                    cur = tree[i].range_max_min_gap(ll, rr)
                    ans = ans if ans > cur else cur
                ac.st(ans)
        return

    @staticmethod
    def abc_186f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc186/tasks/abc186_f
        tag: PointSetAddRangeSum|implemention|brain_teaser|brute_force|contribution_method
        """
        m, n, k = ac.read_list_ints()
        stone = [[] for _ in range(m)]
        col = [m] * n
        row = [n] * m
        for _ in range(k):
            x, y = ac.read_list_ints_minus_one()
            stone[x].append(y)
            col[y] = min(col[y], x)
            row[x] = min(row[x], y)
        ans = 0
        tree = PointSetAddRangeSum(n)
        for y in range(n):
            if col[y] == 0:
                for yy in range(y, n):
                    tree.point_update(yy, (0, 1))
        for x in range(m):
            if row[x] == 0:
                for y in range(n):
                    if col[y] == 0:
                        ans += (n - y) * (m - x)
                        break
                    if col[y] <= x:
                        ans += m - x
                    else:
                        ans += m - col[y]
                break
            if not stone[x]:
                continue
            res = n
            for y in stone[x]:
                tree.point_update(y, (0, 1))
                res = min(res, y)
            ans += tree.range_query(res, n - 1)
        ans = m * n - ans
        ac.st(ans)
        return

    @staticmethod
    def abc_179f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc179/tasks/abc179_f
        tag: zkw_segment_tree|implemention|brain_teaser
        """
        n, q = ac.read_list_ints()
        ans = (n - 2) * (n - 2)
        row = RangeUpdatePointQuery(n + 1, n, min)
        col = RangeUpdatePointQuery(n + 1, n, min)

        for _ in range(q):
            op, x = ac.read_list_ints()
            if op == 1:
                cur = col.point_get(x)
                ans -= cur - 2
                row.range_merge(1, cur, x)
            else:
                cur = row.point_get(x)
                ans -= cur - 2
                col.range_merge(1, cur, x)
        ac.st(ans)
        return

    @staticmethod
    def abc_178e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc178/tasks/abc178_e
        tag: PointUpdateRangeQuery|manhattan_distance|classical
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nodes = set()
        for x, y in nums:
            nodes.add(y)
        nodes = sorted(nodes)
        ind = {num: i for i, num in enumerate(nodes)}
        m = len(ind)
        inf = 2 * 10 ** 9
        tree_pos = PointUpdateRangeQuery(m, -inf, max)
        tree_neg = PointUpdateRangeQuery(m, -inf, max)
        ans = 0
        nums.sort()
        for x, y in nums:
            ans = max(ans, x + y + tree_pos.range_query(0, ind[y]))
            ans = max(ans, x - y + tree_neg.range_query(ind[y], m - 1))
            tree_pos.point_update(ind[y], -x - y)
            tree_neg.point_update(ind[y], -x + y)
        ac.st(ans)
        return

    @staticmethod
    def abc_379d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc379/tasks/abc379_d
        tag: RangeAddPointGet|diff_array|classical
        """
        q = ac.read_int()
        stack = []
        ind = 0
        tree = RangeAddPointGet(q)
        for i in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                stack.append(i)
            elif lst[0] == 2:
                t = lst[1]
                tree.range_add(0, i, t)
            else:
                h = lst[1]
                ans = 0
                while ind < len(stack) and tree.point_get(stack[ind]) >= h:
                    ind += 1
                    ans += 1
                ac.st(ans)
        return

    @staticmethod
    def abc_382f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc382/tasks/abc382_f
        tag: RangeDescendRangeMin|implemention
        """
        m, n, k = ac.read_list_ints()
        nums = [ac.read_list_ints_minus_one() for _ in range(k)]
        tree = RangeDescendRangeMin(n, m + 1)
        vals = [nums[i][0] * k + i for i in range(k)]
        vals.sort(reverse=True)
        ans = [0] * k
        for val in vals:
            i = val % k
            rr, cc, ll = nums[i]
            ans[i] = tree.range_min(cc, cc + ll) - 1
            tree.range_descend(cc, cc + ll, ans[i])
        for a in ans:
            ac.st(a)
        return

    @staticmethod
    def lc_3382(x_coord: List[int], y_coord: List[int]) -> int:
        """
        url: https://leetcode.com/problems/maximum-area-rectangle-with-point-constraints-ii/description/
        tag: PointUpdateRangeQuery
        """
        points = [(x, y) for x, y in zip(x_coord, y_coord)]

        nodes = set()
        for x, y in points:
            nodes.add(x)
            nodes.add(y)
        nodes = sorted(set(nodes))
        m = len(nodes)
        ind = {num: i for i, num in enumerate(nodes)}
        dct = defaultdict(list)
        for x, y in points:
            dct[x].append(y)
        inf = 10 ** 9

        tree = PointSetRangeMinCount(m, (inf, 1))
        low = [inf] * m
        ans = -1
        for x in sorted(dct, reverse=True):
            lst = dct[x]
            lst.sort()
            k = len(lst)
            for i in range(k - 1):
                y1, y2 = lst[i], lst[i + 1]
                floor, cnt = tree.range_query(ind[y1], ind[y2])
                if floor == low[ind[y1]] == low[ind[y2]] < inf and cnt == 2:
                    ans = max(ans, (y2 - y1) * (floor - x))
            for y in lst:
                low[ind[y]] = x
                tree.point_update(ind[y], (x, 1))
        return ans
