"""
Algorithm：segment_tree|bisect_left
Description：range_sum|range_min|range_add|range_change|range_max|dynamic_segment_tree|defaulted_dict

====================================LeetCode====================================

=====================================LuoGu======================================

===================================CodeForces===================================
1093G（https://codeforces.com/contest/1093/problem/G）manhattan_distance|point_set|range_max_min_gap|classical

====================================AtCoder=====================================
ABC186F（https://atcoder.jp/contests/abc186/tasks/abc186_f）PointSetPointAddRangeSum|implemention|brain_teaser|brute_force|contribution_method
ABC179F（https://atcoder.jp/contests/abc179/tasks/abc179_f）zkw_segment_tree|implemention|brain_teaser

=====================================AcWing=====================================

=====================================LibraryChecker=====================================


"""
import math

from src.structure.segment_tree.template import PointSetRangeMaxMinGap
from src.structure.zkw_segment_tree.template import PointSetPointAddRangeSum, RangeMergePointGet
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
        tag: PointSetPointAddRangeSum|implemention|brain_teaser|brute_force|contribution_method
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
        tree = PointSetPointAddRangeSum(n)
        for y in range(n):
            if col[y] == 0:
                for yy in range(y, n):
                    tree.point_set(yy, 1)
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
                tree.point_set(y, 1)
                res = min(res, y)
            ans += tree.range_sum(res, n - 1)
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
        row = RangeMergePointGet(n + 1, n, min)
        col = RangeMergePointGet(n + 1, n, min)

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
