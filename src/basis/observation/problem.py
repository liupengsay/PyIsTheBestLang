"""

Algorithm：observation
Description：observation|property|data_range

====================================LeetCode====================================

=====================================LuoGu======================================

===================================CodeForces===================================
1305C（https://codeforces.com/problemset/problem/1305/C）observation|property|data_range
1705D（https://codeforces.com/problemset/problem/1705/D）observation|property
1646C（https://codeforces.com/problemset/problem/1646/C）observation|data_range
1809D（https://codeforces.com/problemset/problem/1809/D）observation|data_range|construction
1749D（https://codeforces.com/problemset/problem/1749/D）observation|data_range
1185C2（https://codeforces.com/problemset/problem/1185/C2）data_range|bucket_cnt|greedy

====================================AtCoder=====================================




"""
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1185c2(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1185/C2
        tag: data_range|bucket_cnt|greedy
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        cnt = [0] * 101
        ans = []
        pre = 0
        for num in nums:
            rest = m - pre
            cur = 0
            for x in range(100, 0, -1):
                if rest >= num:
                    break
                y = min(cnt[x], (num - rest + x - 1) // x)
                cur += y
                rest += y * x
            ans.append(cur)
            cnt[num] += 1
            pre += num
        ac.lst(ans)
        return
