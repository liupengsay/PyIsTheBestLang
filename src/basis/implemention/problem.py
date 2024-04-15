"""
Algorithm：implemention|big_implemention
Description：implemention|joseph_circle

====================================LeetCode====================================
2296（https://leetcode.cn/problems/design-a-text-editor/）pointer|implemention
54（https://leetcode.cn/problems/spiral-matrix/）num_to_pos|pos_to_num|matrix_spiral
59（https://leetcode.cn/problems/spiral-matrix-ii/）num_to_pos|pos_to_num|matrix_spiral
2326（https://leetcode.cn/problems/spiral-matrix-iv/）num_to_pos|pos_to_num|matrix_spiral
62（https://leetcode.cn/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/）joseph_circle|
2534（https://leetcode.cn/problems/time-taken-to-cross-the-door/）implemention
460（https://leetcode.cn/problems/lfu-cache/）order_dict|lfu
146（https://leetcode.cn/problems/lru-cache/）order_dict|lru
2534（https://leetcode.cn/problems/time-taken-to-cross-the-door/）implemention
1823（https://leetcode.cn/problems/find-the-winner-of-the-circular-game/）joseph_circle
927（https://leetcode.cn/problems/three-equal-parts/description/）base|bin|implemention
1599（https://leetcode.cn/problems/maximum-profit-of-operating-a-centennial-wheel/）implemention|brute_force
2295（https://leetcode.cn/problems/replace-elements-in-an-array/description/）reverse_thinking|linked_list
1914（https://leetcode.cn/problems/cyclically-rotating-a-grid/description/）pointer|circular|implemention
1834（https://leetcode.cn/contest/weekly-contest-237/problems/single-threaded-cpu/）heapq|pointer|implemention

=====================================LuoGu======================================
P1815（https://www.luogu.com.cn/problem/P1815）implemention
P1538（https://www.luogu.com.cn/problem/P1538）implemention
P1535（https://www.luogu.com.cn/problem/P1535）dp|implemention|counter
P2239（https://www.luogu.com.cn/problem/P2239）implemention|matrix_spiral
P2338（https://www.luogu.com.cn/problem/P2338）implemention
P2366（https://www.luogu.com.cn/problem/P2366）implemention
P2552（https://www.luogu.com.cn/problem/P2552）implemention
P2696（https://www.luogu.com.cn/problem/P2696）joseph_circle|implemention|diff_array
P1234（https://www.luogu.com.cn/problem/P1234）implemention
P1166（https://www.luogu.com.cn/problem/P1166）implemention
P1076（https://www.luogu.com.cn/problem/P1076）implemention
P8924（https://www.luogu.com.cn/problem/P8924）implemention|base
P8889（https://www.luogu.com.cn/problem/P8889）counter
P8870（https://www.luogu.com.cn/problem/P8870）base|implemention
P3880（https://www.luogu.com.cn/problem/P3880）implemention
P3111（https://www.luogu.com.cn/problem/P3111）reverse_thinking|implemention
P4346（https://www.luogu.com.cn/problem/P4346）implemention
P5079（https://www.luogu.com.cn/problem/P5079）string|implemention
P5483（https://www.luogu.com.cn/problem/P5483）implemention
P5587（https://www.luogu.com.cn/problem/P5587）implemention
P5759（https://www.luogu.com.cn/problem/P5759）implemention|high_precision|division_to_multiplication
P5989（https://www.luogu.com.cn/problem/P5989）implemention|counter
P5995（https://www.luogu.com.cn/problem/P5995）implemention
P6264（https://www.luogu.com.cn/problem/P6264）implemention
P6282（https://www.luogu.com.cn/problem/P6282）reverse_thinking|reverse_order|implemention
P6410（https://www.luogu.com.cn/problem/P6410）implemention
P6480（https://www.luogu.com.cn/problem/P6480）implemention|counter
P7186（https://www.luogu.com.cn/problem/P7186）brain_teaser|action_scope|implemention
P7338（https://www.luogu.com.cn/problem/P7338）greedy|implemention
P2129（https://www.luogu.com.cn/problem/P2129）stack|pointer|implemention
P3407（https://www.luogu.com.cn/problem/P3407）implemention
P5329（https://www.luogu.com.cn/problem/P5329）lexicographical_order|lexicographical_order|sorting
P6397（https://www.luogu.com.cn/problem/P6397）greedy|implemention
P8247（https://www.luogu.com.cn/problem/P8247）implemention
P8611（https://www.luogu.com.cn/problem/P8611）implemention|classification_discussion
P8755（https://www.luogu.com.cn/problem/P8755）heapq|implemention
P9023（https://www.luogu.com.cn/problem/P9023）matrix_rotate|implemention|counter
P8898（https://www.luogu.com.cn/problem/P8898）greedy|implemention
P8895（https://www.luogu.com.cn/problem/P8895）implemention|counter
P8884（https://www.luogu.com.cn/problem/P8884）classification_discussion|odd_even
P8873（https://www.luogu.com.cn/problem/P8873）math|arithmetic_sequence

===================================CodeForces===================================
463C（https://codeforces.com/problemset/problem/463/C）diagonal|matrix
1676D（https://codeforces.com/contest/1676/problem/D）skill|diagonal|matrix
1703E（https://codeforces.com/contest/1703/problem/E）matrix|rotate
1722F（https://codeforces.com/contest/1722/problem/F）
1807F（https://codeforces.com/contest/1807/problem/F）implemention|classical
1850G（https://codeforces.com/contest/1850/problem/G）implemention|classical|matrix_direction
1006D（https://codeforces.com/contest/1006/problem/D）greedy|implemention|brute_force
1506F（https://codeforces.com/contest/1506/problem/F）implemention|odd_even
1560E（https://codeforces.com/contest/1560/problem/E）reverse_thinking|implemention

====================================AtCoder=====================================
ABC334B（https://atcoder.jp/contests/abc334/tasks/abc334_b）implemention|greedy|brute_force
ABC321E（https://atcoder.jp/contests/abc321/tasks/abc321_e）implemention|binary_tree|counter
ABC315D（https://atcoder.jp/contests/abc315/tasks/abc315_d）bfs|classical|implemention
ABC278D（https://atcoder.jp/contests/abc278/tasks/abc278_d）brain_teaser|classical
ABC279E（https://atcoder.jp/contests/abc279/tasks/abc279_e）prefix_suffix|implemention|brain_teaser|classical
ABC274D（https://atcoder.jp/contests/abc274/tasks/abc274_d）brute_force|implemention
ABC273D（https://atcoder.jp/contests/abc273/tasks/abc273_d）binary_search|implemention
ABC273E（https://atcoder.jp/contests/abc273/tasks/abc273_e）tree|implemention|implemention|classical
ABC272E（https://atcoder.jp/contests/abc272/tasks/abc272_e）brute_force|implemention|euler_series|classical
ABC270B（https://atcoder.jp/contests/abc270/tasks/abc270_b）brute_force|implemention
ABC253G（https://atcoder.jp/contests/abc253/tasks/abc253_g）inclusion_exclusion|prefix_sum|implemention|permutation|classical

=====================================AcWing=====================================
4318（https://www.acwing.com/problem/content/description/4321/）hash|greedy|implemention|construction


"""
import math
from collections import deque
from heapq import heappop, heappush

from src.basis.implemention.template import SpiralMatrix
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_1823(n: int, k: int) -> int:
        """
        url: https://leetcode.cn/problems/find-the-winner-of-the-circular-game/
        tag: joseph_circle
        """
        return SpiralMatrix.joseph_circle(n, k) + 1

    @staticmethod
    def cf_463c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/463/C
        tag: diagonal|matrix|odd_even|brain_teaser
        """
        n = ac.read_int()
        grid = [ac.read_list_ints() for _ in range(n)]
        left = [0] * 2 * n
        right = [0] * 2 * n
        for i in range(n):
            for j in range(n):
                left[i - j] += grid[i][j]
                right[i + j] += grid[i][j]

        ans1 = [-1, -1]
        ans2 = [[-1, -1], [-1, -1]]
        for i in range(n):
            for j in range(n):
                cur = left[i - j] + right[i + j] - grid[i][j]
                t = (i + j) & 1
                if cur > ans1[t]:
                    ans1[t] = cur
                    ans2[t] = [i + 1, j + 1]

        ac.st(sum(ans1))
        ac.lst(ans2[0] + ans2[1])
        return

    @staticmethod
    def lg_p1815(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1815
        tag: implemention
        """

        def check():
            lst = deque([[25, j] for j in range(11, 31)])
            dire = {"E": [0, 1], "S": [1, 0], "W": [0, -1], "N": [-1, 0]}
            m = 0
            for i, w in enumerate(s):
                m = i + 1
                x, y = lst[-1]
                a, b = dire[w]
                x += a
                y += b
                if not (1 <= x <= 50 and 1 <= y <= 50):
                    return f"The worm ran off the board on move {m}."
                if [x, y] in lst and [x, y] != lst[0]:
                    return f"The worm ran into itself on move {m}."
                lst.popleft()
                lst.append([x, y])
            return f"The worm successfully made all {m} moves."

        while True:
            s = ac.read_int()
            if not s:
                break
            s = ac.read_str()
            ac.st(check())
        return

    @staticmethod
    def lg_p2129(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2129
        tag: stack|pointer|implemention
        """
        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        mul = [1, 1]
        add = [0, 0]
        for lst in [ac.read_list_strs() for _ in range(m)][::-1]:
            if lst[0] == "x":
                mul[0] *= -1
                add[0] *= -1
            elif lst[0] == "y":
                mul[1] *= -1
                add[1] *= -1
            else:
                p, q = lst[1:]
                add[0] += int(p)
                add[1] += int(q)
        for x, y in nums:
            ac.lst([mul[0] * x + add[0], mul[1] * y + add[1]])
        return

    @staticmethod
    def lg_p3407(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3407
        tag: implemention
        """
        n, t, q = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        pre = [-inf] * n
        right = -inf
        sta = -inf
        for i in range(n):
            a, r = nums[i]
            if r == 1:
                right = a
                sta = -inf
            else:
                if right != -inf:
                    sta = (right + a) // 2
                    pre[i] = sta
                    right = -inf
                elif sta != -inf:
                    pre[i] = sta

        post = [inf] * n
        left = inf
        sta = inf
        for i in range(n - 1, -1, -1):
            a, r = nums[i]
            if r == 2:
                left = a
                sta = inf
            else:
                if left != inf:
                    sta = (left + a) // 2
                    post[i] = sta
                    left = inf
                elif sta != inf:
                    post[i] = sta

        for _ in range(q):
            i = ac.read_int() - 1
            a, r = nums[i]
            if r == 1:
                ac.st(ac.min(a + t, post[i]))
            else:
                ac.st(ac.max(a - t, pre[i]))
        return

    @staticmethod
    def lg_p5329(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5329
        tag: lexicographical_order|sort|brain_teaser|classical
        """
        n = ac.read_int()
        s = ac.read_str()
        ans = [0] * n
        i, j = 0, n - 1
        idx = 0
        for x in range(1, n):
            if s[x] > s[x - 1]:
                for y in range(x - 1, idx - 1, -1):
                    ans[j] = y + 1
                    j -= 1
                idx = x
            if s[x] < s[x - 1]:
                for y in range(idx, x):
                    ans[i] = y + 1
                    i += 1
                idx = x
        for x in range(idx, n):
            ans[i] = x + 1
            i += 1
        ac.lst(ans)
        return

    @staticmethod
    def lg_p6397(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6397
        tag: greedy|implemention|brain_teaser
        """
        k = ac.read_float()
        nums = [ac.read_float() for _ in range(ac.read_int())]
        pre = nums[0]
        ans = 0
        for num in nums[1:]:
            if num - ans <= pre + k <= num + ans:
                pre += k
            elif pre + k > num + ans:
                pre = ac.max(pre, num + ans)
            else:
                gap = (num - ans - pre - k) / 2.0
                pre = num - ans - gap
                ans += gap
        ac.st(ans)
        return

    @staticmethod
    def lg_p8247(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8247
        tag: implemention|math|line_slope
        """
        m, n = ac.read_list_ints()
        start = [-1, -1]
        dct = []
        for i in range(m):
            lst = ac.read_str()
            for j in range(n):
                if lst[j] == "S":
                    start = [i, j]
                elif lst[j] == "K":
                    dct.append([i, j])
        a, b = start
        cnt = set()
        for i, j in dct:
            x, y = i - a, j - b
            if x == 0:
                y = 1 if y > 0 else -1
            else:
                g = math.gcd(abs(x), abs(y))
                x //= g
                y //= g
            cnt.add((x, y))
        ac.st(len(cnt))
        return

    @staticmethod
    def lg_p8611(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8611
        tag: implemention|classification_discussion|classical|brain_teaser
        """
        ac.read_int()
        nums = ac.read_list_ints()
        a = nums[0]
        x = y = 0
        for num in nums[1:]:
            if abs(num) > abs(a) and num < 0:
                y += 1
            elif abs(num) < abs(a) and num > 0:
                x += 1
        if a < 0:
            ans = 1 if not x else x + y + 1
        else:
            ans = 1 if not y else x + y + 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p9023(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9023
        tag: matrix_rotate|implemention|counter|odd_even|inclusion_exclusion
        """
        m = ac.read_int()
        n = ac.read_int()
        k = ac.read_int()
        row = [0] * (m + 1)
        col = [0] * (n + 1)
        for _ in range(k):
            lst = ac.read_list_strs()
            x = int(lst[1])
            if lst[0] == "R":
                row[x] += 1
                row[x] %= 2
            else:
                col[x] += 1
                col[x] %= 2
        cnt1 = sum(row)
        cnt2 = sum(col)
        ans = cnt1 * (n - cnt2) + cnt2 * (m - cnt1)
        ac.st(ans)
        return

    @staticmethod
    def lg_p8895(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8895
        tag: implemention|counter|classical|freq
        """
        n, m, p = ac.read_list_ints()
        dp = [1] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] * 2 % p

        def check():
            while stack and not cnt[stack[0]]:
                heappop(stack)
            if ceil >= 3 or cnt[stack[0]] != 1:
                ac.st(0)
            else:
                even = freq[2]
                ac.st(dp[n - even * 2 - 1])
            return

        def add(a):
            nonlocal ceil
            heappush(stack, a)
            cnt[a] += 1
            if ceil < cnt[a]:
                ceil = cnt[a]
            freq[cnt[a]] += 1
            if cnt[a] > 1:
                freq[cnt[a] - 1] -= 1
            return

        def remove(a):
            nonlocal ceil
            freq[cnt[a]] -= 1
            if not freq[ceil]:
                ceil -= 1
            cnt[a] -= 1
            if cnt[a]:
                freq[cnt[a]] += 1
            return

        freq = [0] * (n + 1)
        cnt = [0] * (n + 1)
        ceil = 0
        nums = ac.read_list_ints()
        stack = []
        for num in nums:
            add(num)

        check()
        for _ in range(m):
            x, k = ac.read_list_ints()
            x -= 1
            remove(nums[x])
            nums[x] = k
            add(k)
            check()
        return

    @staticmethod
    def lg_p8884(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8884
        tag: classification_discussion|odd_even
        """
        n, m, c = ac.read_list_ints()
        cnt = [0, 0]
        for _ in range(c):
            i, j = ac.read_list_ints_minus_one()
            cnt[(i + j) % 2] += 1

        total = [0, 0]
        if m % 2 == 0 or n % 2 == 0:
            total[0] = total[1] = m * n // 2
        else:
            total[0] = m * n // 2 + 1
            total[1] = m * n // 2

        for _ in range(ac.read_int()):
            x1, y1, x2, y2, p = ac.read_list_ints_minus_one()
            p += 1
            cur = [0, 0]
            while p:
                lst = ac.read_list_ints()
                if not lst:
                    continue
                i, j = [x - 1 for x in lst]
                cur[(i + j) % 2] += 1
                p -= 1

            mm, nn = x2 - x1 + 1, y2 - y1 + 1
            cur_total = [0, 0]
            if (mm * nn) % 2 == 0:
                cur_total[0] = cur_total[1] = mm * nn // 2
            else:
                if (x1 + y1) % 2 == 0:
                    cur_total[0] = mm * nn // 2 + 1
                    cur_total[1] = mm * nn // 2
                else:
                    cur_total[1] = mm * nn // 2 + 1
                    cur_total[0] = mm * nn // 2

            if cur[0] <= cnt[0] and cur[1] <= cnt[1] \
                    and total[0] - cur_total[0] >= cnt[0] - cur[0] \
                    and total[1] - cur_total[1] >= cnt[1] - cur[1]:
                ac.st("YES")
            else:
                ac.st("NO")

        return

    @staticmethod
    def ac_4318(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4321/
        tag: hash|greedy|implemention|construction|brain_teaser
        """
        x = y = 0
        ind = dict()
        ind["U"] = [-1, 0]
        ind["D"] = [1, 0]
        ind["L"] = [0, -1]
        ind["R"] = [0, 1]
        pre = {(0, 0)}
        for w in ac.read_str():
            cur = (x, y)
            x += ind[w][0]
            if (x, y) in pre:
                ac.st("NO")
                return
            for a, b in [[-1, 0], [0, 1], [1, 0], [0, -1]]:
                if (x + a, y + b) in pre and (x + a, y + b) != cur:
                    ac.st("NO")
                    return
            pre.add((x, y))
        ac.st("YES")
        return

    @staticmethod
    def abc_279e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc279/tasks/abc279_e
        tag: prefix_suffix|implemention|brain_teaser|classical
        """
        n, m = ac.read_list_ints()
        a = ac.read_list_ints_minus_one()
        zero = 0
        pre = [zero]
        for i in range(m - 1):
            aa = a[i]
            if aa + 1 == zero:
                zero = aa
            elif aa == zero:
                zero = aa + 1
            pre.append(zero)

        ans = [pre[-1]]
        b = list(range(n))
        for i in range(m - 1, -1, -1):
            if i < m - 1:
                ans.append(b[pre[i]])
            aa = a[i]
            b[aa], b[aa + 1] = b[aa + 1], b[aa]

        for a in ans[::-1]:
            ac.st(a + 1)
        return

    @staticmethod
    def abc_273e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc273/tasks/abc273_e
        tag: tree|implemention|implemention|classical
        """
        q = ac.read_int()
        nums = ["-1"] * (q + 1)
        father = [-1] * (q + 1)
        ind = now = 0
        dct = dict()
        ans = []
        for _ in range(q):
            lst = ac.read_list_strs()
            if lst[0] == "ADD":
                ind += 1
                nums[ind] = lst[1]
                father[ind] = now
                now = ind
            elif lst[0] == "DELETE":
                if now:
                    now = father[now]
            elif lst[0] == "SAVE":
                dct[lst[1]] = now
            else:
                now = dct.get(lst[1], 0)
            ans.append(nums[now] if now else -1)
        ac.lst(ans)
        return

    @staticmethod
    def abc_272e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc272/tasks/abc272_e
        tag: brute_force|implemention|euler_series|classical
        """
        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()
        dct = [set() for _ in range(m + 1)]
        for i in range(n):
            num = nums[i]
            low = max(0, (-num) // (i + 1))
            num += low * (i + 1)
            while low <= m and num <= n:
                dct[low].add(num)
                low += 1
                num += (i + 1)
        for i in range(1, m + 1):
            x = 0
            while x in dct[i]:
                x += 1
            ac.st(x)
        return

    @staticmethod
    def abc_270b(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc270/tasks/abc270_b
        tag: brute_force|implemention
        """
        x, y, z = ac.read_list_ints()
        if x < 0:
            x, y, z = -x, -y, -z
        if y < 0 or y > x:
            ac.st(x)
        elif z > y:
            ac.st(-1)
        else:
            ac.st(abs(z) + abs(x - z))
        return

    @staticmethod
    def abc_253g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc253/tasks/abc253_g
        tag: inclusion_exclusion|prefix_sum|implemention|permutation|classical
        """
        n, ll, rr = ac.read_list_ints()

        def check(x):
            if x == 0:
                return list(range(n))
            pre = 0
            while x >= (n - pre - 1) >= 0:
                x -= (n - pre - 1)
                pre += 1
            nums = list(range(n - 1, n - pre - 1, -1)) + list(range(n - pre))
            if pre < n:
                start = pre
                for y in range(1, x + 1):
                    nums[start], nums[start + y] = nums[start + y], nums[start]
            return nums

        nums1 = check(ll - 1)
        nums2 = check(rr)
        dct = {num: i for i, num in enumerate(nums2)}
        ans = [0] * n
        for i in range(n):
            ans[dct[nums1[i]]] = i
        ac.lst([x + 1 for x in ans])
        return