"""
Algorithm：implemention、implemention、大implemention
Function：根据题意implemention，有implemention结论约瑟夫环问题

====================================LeetCode====================================
2296（https://leetcode.com/problems/design-a-text-editor/）pointer维护结果implemention
54（https://leetcode.com/problems/spiral-matrix/）https://leetcode.com/problems/spiral-matrix/ 两种方式，数字到索引，以及索引到数字
59（https://leetcode.com/problems/spiral-matrix-ii/）
2326（https://leetcode.com/problems/spiral-matrix-iv/）
剑指 Offer 62（https://leetcode.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/）约瑟夫环
2534（https://leetcode.com/problems/time-taken-to-cross-the-door/）根据题意implemention
460（https://leetcode.com/problems/lfu-cache/）OrderDict应用与数据结构自定义题目
146（https://leetcode.com/problems/lru-cache/）OrderDict应用与数据结构自定义题目
2534（https://leetcode.com/problems/time-taken-to-cross-the-door/）按照时间与题意implemention
1823（https://leetcode.com/contest/weekly-contest-236/problems/find-the-winner-of-the-circular-game/）约瑟夫环
927（https://leetcode.com/problems/three-equal-parts/description/）根据二进制特点确定三部分是否相等
1599（https://leetcode.com/problems/maximum-profit-of-operating-a-centennial-wheel/description/）典型implemention题brute_force
2295（https://leetcode.com/problems/replace-elements-in-an-array/description/）逆序思维，或者类似链表思想
1914（https://leetcode.com/problems/cyclically-rotating-a-grid/description/）pointer循环implemention
1834（https://leetcode.com/contest/weekly-contest-237/problems/single-threaded-cpu/）堆和pointerimplemention

=====================================LuoGu======================================
1815（https://www.luogu.com.cn/problem/P1815）implemention类似贪吃蛇的移动
1538（https://www.luogu.com.cn/problem/P1538）implemention数字文本的打印
1535（https://www.luogu.com.cn/problem/P1535）动态规划implementioncounter
2239（https://www.luogu.com.cn/problem/P2239）implementionmatrix_spiral的赋值
2338（https://www.luogu.com.cn/problem/P2338）按照题意时间与距离的implemention
2366（https://www.luogu.com.cn/problem/P2366）字符串implemention与变量赋值
2552（https://www.luogu.com.cn/problem/P2552）矩阵赋值implemention
2696（https://www.luogu.com.cn/problem/P2696）约瑟夫环implemention与差分
1234（https://www.luogu.com.cn/problem/P1234）矩阵每个点四个方向特定长为4的单词个数
1166（https://www.luogu.com.cn/problem/P1166）按照题意复杂的implemention题
1076（https://www.luogu.com.cn/problem/P1076）implemention操作即可
8924（https://www.luogu.com.cn/problem/P8924）implemention的同时进制的思想求解
8889（https://www.luogu.com.cn/problem/P8889）01序列分段counter
8870（https://www.luogu.com.cn/problem/P8870）按照进制|法implemention
3880（https://www.luogu.com.cn/problem/P3880）按照题意implemention|密字符串
3111（https://www.luogu.com.cn/problem/P3111）reverse_thinking行进距离implemention分组，类似力扣车队题目
4346（https://www.luogu.com.cn/problem/P4346）implemention数字与字符串的转换与|减
5079（https://www.luogu.com.cn/problem/P5079）字符串implemention
5483（https://www.luogu.com.cn/problem/P5483）implemention表格拼接
5587（https://www.luogu.com.cn/problem/P5587）按照题意implemention统计
5759（https://www.luogu.com.cn/problem/P5759）按照题意implemention统计，将除法转换为乘法避免引起精度问题的处理技巧
5989（https://www.luogu.com.cn/problem/P5989）implementioncounter确定每个点左右角上方移除点的个数
5995（https://www.luogu.com.cn/problem/P5995）动态implemention更新结果
6264（https://www.luogu.com.cn/problem/P6264）implemention和循环判断
6282（https://www.luogu.com.cn/problem/P6282）reverse_thinking，倒序分配implemention
6410（https://www.luogu.com.cn/problem/P6410）按照题意和数独问题implemention
6480（https://www.luogu.com.cn/problem/P6480）implemention摆放位置counter
7186（https://www.luogu.com.cn/problem/P7186）brain_teaser，有限数据与作用域implemention
7338（https://www.luogu.com.cn/problem/P7338）greedyimplemention赋值
2129（https://www.luogu.com.cn/problem/P2129）栈和pointerimplemention
3407（https://www.luogu.com.cn/problem/P3407）implemention移动与相遇
5329（https://www.luogu.com.cn/problem/P5329）lexicographical_order应用题，依据相邻项的lexicographical_order大小来确认sorting
6397（https://www.luogu.com.cn/problem/P6397）greedyimplemention
8247（https://www.luogu.com.cn/problem/P8247）implemention按照相对位置比例区分
8611（https://www.luogu.com.cn/problem/P8611）蚂蚁碰撞implementionclassification_discussion
8755（https://www.luogu.com.cn/problem/P8755）二叉堆implemention
9023（https://www.luogu.com.cn/problem/P9023）矩阵翻转implementioncounter
8898（https://www.luogu.com.cn/problem/P8898）greedyimplemention
8895（https://www.luogu.com.cn/problem/P8895）implemention与组合counter
8884（https://www.luogu.com.cn/problem/P8884）分矩阵位置的奇偶性讨论
8873（https://www.luogu.com.cn/problem/P8873）等差数列

===================================CodeForces===================================
463C（https://codeforces.com/problemset/problem/463/C）选取两组互不相交的主副对角线使得和最大

=====================================AcWing=====================================
4318（https://www.acwing.com/problem/content/description/4321/）hashgreedyimplemention构造


"""
import heapq
import math
from collections import deque, Counter

from src.basis.implemention.template import SpiralMatrix
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_1823(n: int, m: int) -> int:
        # 模板: 约瑟夫环最后的幸存者
        return SpiralMatrix.joseph_ring(n, m) + 1

    @staticmethod
    def cf_463c(ac=FastIO()):
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
                # 两个主教的位置，坐标和分别为一个奇数一个偶数才不会相交
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
        # 根据指令方格组合移动
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
        # 栈和pointerimplemention
        n, m = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        lst_x = []
        lst_y = []
        cnt_x = cnt_y = 0
        for lst in [ac.read_list_strs() for _ in range(m)][::-1]:
            if lst[0] == "x":
                cnt_x += 1
            elif lst[0] == "y":
                cnt_y += 1
            else:
                p, q = lst[1:]
                lst_x.append([int(p), cnt_x])
                lst_y.append([int(q), cnt_y])
        add_x = add_y = 0
        for a, b in lst_x:
            diff = cnt_x - b
            add_x += a if diff % 2 == 0 else -a

        for a, b in lst_y:
            diff = cnt_y - b
            add_y += a if diff % 2 == 0 else -a

        cnt_x %= 2
        cnt_y %= 2
        for a, b in nums:
            a = a if not cnt_x else -a
            b = b if not cnt_y else -b
            ac.lst([a + add_x, b + add_y])
        return

    @staticmethod
    def lg_p3407(ac=FastIO()):
        # implemention相向而行
        n, t, q = ac.read_list_ints()
        nums = deque([ac.read_list_ints() for _ in range(n)])
        pos = [-1] * n

        # 先去除头尾永不相见的
        ind = deque(list(range(n)))
        while ind and nums[ind[0]][1] == 2:
            i = ind.popleft()
            x = nums[i][0]
            pos[i] = x - t
        while ind and nums[ind[-1]][1] == 1:
            i = ind.pop()
            x = nums[i][0]
            pos[i] = x + t

        # 对相向而行的区间段是否到达相遇点
        while ind:
            left = []
            while ind and nums[ind[0]][1] == 1:
                left.append(ind.popleft())
            right = []
            while ind and nums[ind[0]][1] == 2:
                right.append(ind.popleft())
            mid = (nums[right[0]][0] + nums[left[-1]][0]) // 2
            for i in left:
                pos[i] = ac.min(mid, nums[i][0] + t)
            for i in right:
                pos[i] = ac.max(mid, nums[i][0] - t)
        for _ in range(q):
            ac.st(pos[ac.read_int() - 1])
        return

    @staticmethod
    def lg_p5329(ac=FastIO()):
        # lexicographical_order应用题，依据相邻项的lexicographical_order大小来确认sorting
        n = ac.read_int()
        s = ac.read_str()
        ans = [0] * n
        i, j = 0, n - 1
        idx = 0
        for x in range(1, n):
            if s[x] > s[x - 1]:
                # 前面的直接扔到后面必然是最大的（去掉小的s[x-1]）
                for y in range(x - 1, idx - 1, -1):
                    ans[j] = y + 1
                    j -= 1
                idx = x
            if s[x] < s[x - 1]:
                # 前面的直接扔到前面必然是最小的（去掉大的s[x-1]）
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
        # greedyimplemention
        k = ac.read_float()
        nums = [ac.read_float() for _ in range(ac.read_int())]
        pre = nums[0]
        ans = 0
        for num in nums[1:]:
            # 记录当前位置与当前耗时
            if num - ans <= pre + k <= num + ans:  # 直接通知到位修改位置不增|时间
                pre += k
            elif pre + k > num + ans:  # 直接通知到位不增|时间位置受限
                pre = ac.max(pre, num + ans)
            else:  # 需要相向而行花费时间
                gap = (num - ans - pre - k) / 2.0
                pre = num - ans - gap
                ans += gap
        ac.st(ans)
        return

    @staticmethod
    def lg_p8247(ac=FastIO()):
        # implemention按照相对位置比例区分
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
        # 蚂蚁碰撞implementionclassification_discussion
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
        # 矩阵翻转implementioncounter
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
        # implemention与组合counter
        n, m, p = ac.read_list_ints()
        dp = [1] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] * 2 % p
        nums = ac.read_list_ints()
        cnt = Counter(nums)
        stack = nums[:]
        heapq.heapify(stack)
        low = stack[0]
        one = 0
        even = 0
        for num in cnt:
            if cnt[num] == 2:
                even += 1
            else:
                one += 1
        if cnt[low] > 1 or even * 2 + one < n:
            ac.st(0)
        else:
            ac.st(dp[n - even * 2 - 1])
        for _ in range(m):
            x, k = ac.read_list_ints()
            x -= 1
            cnt[nums[x]] -= 1
            if cnt[nums[x]] == 1:
                even -= 1
                one += 1
            elif cnt[nums[x]] == 0:
                one -= 1

            nums[x] = k
            cnt[nums[x]] += 1
            if cnt[nums[x]] == 2:
                even += 1
                one -= 1
            elif cnt[nums[x]] == 1:
                one += 1
            heapq.heappush(stack, k)
            while not cnt[stack[0]]:
                heapq.heappop(stack)
            if cnt[stack[0]] > 1 or even * 2 + one < n:
                ac.st(0)
            else:
                ac.st(dp[n - even * 2 - 1])
        return

    @staticmethod
    def lg_p8884(ac=FastIO()):
        # 分矩阵位置的奇偶性讨论
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
        # hashgreedyimplemention构造
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
            y += ind[w][1]
            # 先前走过
            if (x, y) in pre:
                ac.st("NO")
                return
            # 先前走过附近的
            for a, b in [[-1, 0], [0, 1], [1, 0], [0, -1]]:
                if (x + a, y + b) in pre and (x + a, y + b) != cur:
                    ac.st("NO")
                    return
            pre.add((x, y))
        ac.st("YES")
        return