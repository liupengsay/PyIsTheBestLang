"""
Algorithm：segment_tree|bisect_left
Description：range_sum|range_min|range_add|range_change|range_max|dynamic_segment_tree|defaulteddict

====================================LeetCode====================================

=====================================LuoGu======================================

===================================CodeForces===================================
1093G（https://codeforces.com/contest/1093/problem/G）manhattan_distance|point_set|range_max_min_gap|classical

====================================AtCoder=====================================

=====================================AcWing=====================================

=====================================LibraryChecker=====================================


"""
from src.data_structure.segment_tree.template import PointSetRangeMaxMinGap
from src.utils.fast_io import FastIO



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
                ans = -inf
                ll, rr = [w - 1 for w in lst[1:]]
                for i in range(m):
                    cur = tree[i].range_max_min_gap(ll, rr)
                    ans = ans if ans > cur else cur
                ac.st(ans)
        return
