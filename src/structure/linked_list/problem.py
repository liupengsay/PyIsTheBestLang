"""

Algorithm：linked_list|double_linked_list|union_find_right_root|union_left
Description：


====================================LeetCode====================================
2617（https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/）bfs|double_linked_list
2612（https://leetcode.cn/problems/minimum-reverse-operations/）bfs|double_linked_list
1562（https://leetcode.cn/problems/find-latest-group-of-size-m/）union_find|double_linked_list
2382（https://leetcode.cn/problems/maximum-segment-sum-after-removals/）reverse_thinking|double_linked_list|union_find
2289（https://leetcode.cn/problems/steps-to-make-array-non-decreasing/description/）monotonic_stack|liner_dp|bfs|linked_list

====================================NewCoder===================================
49888C（https://ac.nowcoder.com/acm/contest/49888/C）double_linked_list

=====================================LuoGu======================================
P5462（https://www.luogu.com.cn/problem/P5462）double_linked_list|greed|lexicographical_order|deque
P6155（https://www.luogu.com.cn/problem/P6155）sort|greed|union_find_right_root

===================================CodeForces===================================
1154E（https://codeforces.com/contest/1154/problem/E）double_linked_list|union_find_left|union_find_right
1922D（https://codeforces.com/problemset/problem/1922/D）double_linked_list|bfs|classical|brain_teaser|implemention

====================================AtCoder=====================================
ABC344E（https://atcoder.jp/contests/abc344/tasks/abc344_e）linked_list
ABC237D（https://atcoder.jp/contests/abc237/tasks/abc237_d）double_linked_list|implemention|classical
ABC225D（https://atcoder.jp/contests/abc225/tasks/abc225_d）linked_list

=====================================AcWing=====================================
136（https://www.acwing.com/problem/content/138/）linked_list|reverse_thinking
4943（https://www.acwing.com/problem/content/description/4946/）bfs|linked_list
5034（https://www.acwing.com/problem/content/5037/）heapq|greed|linked_list

"""
import heapq
import math
from collections import deque
from typing import List

from src.graph.union_find.template import UnionFind
from src.util.fast_io import FastIO



class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1154e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1154/problem/E
        tag: double_linked_list
        """
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        ans = [0] * n
        pre = [i - 1 for i in range(n)]
        nex = [i + 1 for i in range(n)]
        ind = [0] * n
        for i in range(n):
            ind[nums[i] - 1] = i

        step = 1
        for num in range(n - 1, -1, -1):
            i = ind[num]
            if ans[i]:
                continue
            ans[i] = step
            left, right = pre[i], nex[i]
            for _ in range(k):
                if left != -1:
                    ans[left] = step
                    left = pre[left]
                else:
                    break
            for _ in range(k):
                if right != n:
                    ans[right] = step
                    right = nex[right]
                else:
                    break
            if left >= 0:
                nex[left] = right
            if right < n:
                pre[right] = left
            step = 3 - step
        ac.st("".join(str(x) for x in ans))
        return

    @staticmethod
    def nc_49888c_1(ac=FastIO()):
        """
        url: https://ac.nowcoder.com/acm/contest/49888/C
        tag: double_linked_list
        """
        n, k = ac.read_list_ints()
        pre = list(range(-1, n + 1))
        post = list(range(1, n + 3))
        assert len(pre) == len(post) == n + 2
        for _ in range(k):
            op, x = ac.read_list_ints()
            if op == 1:
                a = pre[x]
                b = post[x]
                if 1 <= b <= n:
                    pre[b] = a
                if 1 <= a <= n:
                    post[a] = b
                pre[x] = post[x] = -1
            else:
                ac.st(pre[x])
        return

    @staticmethod
    def nc_49888c_2(ac=FastIO()):
        """
        url: https://ac.nowcoder.com/acm/contest/49888/C
        tag: double_linked_list|union_find_root_left
        """
        n, k = ac.read_list_ints()
        uf = UnionFind(n + 1)
        for _ in range(k):
            op, x = ac.read_list_ints()
            if op == 1:
                uf.union(x - 1, x)
            else:
                ac.st(uf.find(x - 1))
        return

    @staticmethod
    def lc_2289(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/steps-to-make-array-non-decreasing/description/
        tag: monotonic_stack|liner_dp|bfs|linked_list
        """
        n = len(nums)
        post = list(range(1, n + 1))
        nums.append(10 ** 9 + 7)
        stack = [i - 1 for i in range(1, n) if nums[i - 1] > nums[i]]
        ans = 0
        visit = [0] * n
        while stack:
            ans += 1
            for i in stack[::-1]:
                if nums[i] > nums[post[i]]:
                    visit[post[i]] = 1
                    post[i] = post[post[i]]
            stack = [i for i in stack if not visit[i] and nums[i] > nums[post[i]]]
        return ans

    @staticmethod
    def lc_2617_1(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/
        tag: bfs|double_linked_list|classical|reverse_order|reverse_thinking|tree_array|segment_tree|flatten
        """
        m, n = len(grid), len(grid[0])
        dis = [[math.inf] * n for _ in range(m)]
        row_nex = [list(range(1, n + 1)) for _ in range(m)]
        col_nex = [list(range(1, m + 1)) for _ in range(n)]
        stack = deque([[0, 0]])
        dis[0][0] = 1

        while stack:
            i, j = stack.popleft()
            d = dis[i][j]
            x = grid[i][j]
            if x == 0:
                continue

            nex = row_nex[i]
            y = nex[j]
            lst = []
            while y <= j + x and 0 <= y < n:
                if dis[i][y] == math.inf:
                    dis[i][y] = d + 1
                    if i == m - 1 and y == n - 1:
                        return d + 1
                    stack.append([i, y])
                lst.append(y)
                y = nex[y]
            for w in lst:
                nex[w] = y

            nex = col_nex[j]
            y = nex[i]
            lst = []
            while y <= i + x and 0 <= y < m:
                if dis[y][j] == math.inf:
                    dis[y][j] = d + 1
                    if y == m - 1 and j == n - 1:
                        return d + 1
                    stack.append([y, j])
                lst.append(y)
                y = nex[y]
            for w in lst:
                nex[w] = y

        ans = dis[-1][-1]
        return ans if ans < math.inf else -1

    @staticmethod
    def lc_2617_2(grid: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/
        tag: bfs|double_linked_list|classical|reverse_order|reverse_thinking|tree_array|segment_tree|flatten
        """
        m, n = len(grid), len(grid[0])
        dis = [[math.inf] * n for _ in range(m)]
        row = [UnionFind(n + 1) for _ in range(m)]
        col = [UnionFind(m + 1) for _ in range(n)]
        stack = deque([[0, 0]])
        dis[0][0] = 1

        while stack:
            i, j = stack.popleft()
            d = dis[i][j]
            x = grid[i][j]
            if x == 0:
                continue

            uf = row[i]
            p = uf.find(j)
            while p <= x + j and p <= n - 1:
                if dis[i][p] == math.inf:
                    dis[i][p] = d + 1
                    stack.append([i, p])
                    if i == m - 1 and p == n - 1:
                        return d + 1
                uf.union(p, p + 1)
                p = uf.find(p)

            uf = col[j]
            p = uf.find(i)
            while p <= x + i and p <= m - 1:
                if dis[p][j] == math.inf:
                    dis[p][j] = d + 1
                    stack.append([p, j])
                    if p == m - 1 and j == n - 1:
                        return d + 1
                uf.union(p, p + 1)
                p = uf.find(p)
        ans = dis[-1][-1]
        return ans if ans < math.inf else -1

    @staticmethod
    def ac_136(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/138/
        tag: linked_list|reverse_thinking
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        ind = list(range(n))
        ind.sort(key=lambda it: nums[it])
        dct = {nums[i]: i for i in range(n)}
        pre = [-1] * n
        post = [-1] * n
        for i in range(1, n):
            a, b = ind[i - 1], ind[i]
            post[a] = b
            pre[b] = a
        ans = []
        for x in range(n - 1, 0, -1):
            num = nums[x]
            i = dct[num]
            a = pre[i]
            b = post[i]
            if a != -1 and b != -1:
                if abs(num - nums[a]) < abs(num - nums[b]) or (
                        abs(num - nums[a]) == abs(num - nums[b]) and nums[a] < nums[b]):
                    ans.append([abs(num - nums[a]), a + 1])
                else:
                    ans.append([abs(num - nums[b]), b + 1])
                post[a] = b
                pre[b] = a
            elif a != -1:
                ans.append([abs(num - nums[a]), a + 1])
                post[a] = post[i]
            else:
                ans.append([abs(num - nums[b]), b + 1])
                pre[b] = pre[i]
        for i in range(n - 2, -1, -1):
            ac.lst(ans[i])
        return

    @staticmethod
    def lg_p5462(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5462
        tag: double_linked_list|greed|lexicographical_order|deque|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        pre = [-1] * (n + 1)
        post = [-1] * (n + 1)
        for i in range(n):
            if i:
                pre[nums[i]] = nums[i - 1]
            if i + 1 < n:
                post[nums[i]] = nums[i + 1]

        visit = [0] * (n + 1)
        ans = []
        for num in range(n, 0, -1):
            if visit[num] or post[num] == -1:
                continue
            visit[num] = visit[post[num]] = 1
            ans.extend([num, post[num]])
            x, y = pre[num], post[post[num]]
            if x != -1:
                post[x] = y
            if y != -1:
                pre[y] = x
        ac.lst(ans)
        return

    @staticmethod
    def lg_p6155(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6155
        tag: sort|greed|union_find_right_root|classical
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        a.sort(reverse=True)
        mod = 2 ** 64
        b.sort()
        cnt = []
        pre = {}
        for num in a:
            y = num
            lst = [num]
            while y in pre:
                lst.append(y)
                y = pre[y]
            lst.append(y)
            for x in lst:
                pre[x] = y + 1
            cnt.append(y - num)
        cnt.sort(reverse=True)
        ans = sum(cnt[i] * b[i] for i in range(n))
        ac.st(ans % mod)
        return

    @staticmethod
    def lc_1562(arr: List[int], m: int) -> int:
        """
        url: https://leetcode.cn/problems/find-latest-group-of-size-m/
        tag: union_find|double_linked_list
        """
        n = len(arr)
        left = [-1] * n
        right = [-1] * n
        cnt = [0] * (n + 1)
        ans = -1
        for x, i in enumerate(arr):
            i -= 1
            if i - 1 >= 0 and left[i - 1] != -1:
                if i + 1 < n and right[i + 1] != -1:
                    start = left[i - 1]
                    end = right[i + 1]
                    cnt[i - start] -= 1
                    cnt[end - i] -= 1
                    cnt[end - start + 1] += 1
                    right[start] = end
                    left[end] = start
                else:
                    start = left[i - 1]
                    cnt[i - start] -= 1
                    end = i
                    cnt[end - start + 1] += 1
                    right[start] = end
                    left[end] = start
            elif i + 1 < n and right[i + 1] != -1:
                start = i
                end = right[i + 1]
                cnt[end - i] -= 1
                cnt[end - start + 1] += 1
                right[start] = end
                left[end] = start
            else:
                left[i] = right[i] = i
                cnt[1] += 1

            if cnt[m]:
                ans = x + 1
        return ans

    @staticmethod
    def ac_4943(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4946/
        tag: bfs|linked_list|classical
        """
        m, n, k = ac.read_list_ints()
        grid = [ac.read_str() for _ in range(m)]
        x1, y1, x2, y2 = ac.read_list_ints_minus_one()

        dis = [[math.inf] * n for _ in range(m)]
        row_nex = [list(range(1, n + 1)) for _ in range(m)]
        row_pre = [list(range(-1, n - 1)) for _ in range(m)]
        col_nex = [list(range(1, m + 1)) for _ in range(n)]
        col_pre = [list(range(-1, m - 1)) for _ in range(n)]
        stack = deque([[x1, y1]])
        dis[x1][y1] = 0

        while stack:
            i, j = stack.popleft()
            d = dis[i][j]
            pre = row_pre[i]
            nex = row_nex[i]
            y = nex[j]
            lst = []
            while y <= j + k and 0 <= y < n and grid[i][y] == ".":
                if dis[i][y] == math.inf:
                    dis[i][y] = d + 1
                    stack.append([i, y])
                lst.append(y)
                y = nex[y]
            for w in lst:
                nex[w] = y
                pre[w] = pre[j]

            pre = row_pre[i]
            nex = row_nex[i]
            y = pre[j]
            lst = []
            while j - y <= k and 0 <= y < n and grid[i][y] == ".":
                if dis[i][y] == math.inf:
                    dis[i][y] = d + 1
                    stack.append([i, y])
                lst.append(y)
                y = pre[y]
            for w in lst:
                pre[w] = y
                nex[w] = nex[j]

            pre = col_pre[j]
            nex = col_nex[j]
            y = nex[i]
            lst = []
            while y <= i + k and 0 <= y < m and grid[y][j] == ".":
                if dis[y][j] == math.inf:
                    dis[y][j] = d + 1
                    stack.append([y, j])
                lst.append(y)
                y = nex[y]
            for w in lst:
                nex[w] = y
                pre[w] = pre[i]

            pre = col_pre[j]
            nex = col_nex[j]
            y = pre[i]
            lst = []
            while i - y <= k and 0 <= y < m and grid[y][j] == ".":
                if dis[y][j] == math.inf:
                    dis[y][j] = d + 1
                    stack.append([y, j])
                lst.append(y)
                y = pre[y]
            for w in lst:
                pre[w] = y
                nex[w] = nex[i]

        ans = dis[x2][y2]
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def ac_5034(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/5037/
        tag: heapq|greed|linked_list
        """
        n = ac.read_int()
        s = ac.read_str()
        nums = ac.read_list_ints()
        post = list(range(1, n + 1))
        pre = list(range(-1, n - 1))
        stack = []
        for i in range(n - 1):
            if s[i] != s[i + 1]:
                heapq.heappush(stack, [abs(nums[i + 1] - nums[i]), i, i + 1])

        ans = []
        visit = [0] * n
        while stack:
            _, i, j = heapq.heappop(stack)
            if not visit[i] and not visit[j]:
                visit[i] = 1
                visit[j] = 1
                ans.append([i + 1, j + 1])

                y = post[j]
                while y < n and visit[y]:
                    y = post[y]

                x = pre[i]
                while x >= 0 and visit[x]:
                    x = pre[x]
                if x != -1:
                    post[x] = y
                if y != n:
                    pre[y] = x
                if x != -1 and y != n and s[x] != s[y]:
                    heapq.heappush(stack, [abs(nums[x] - nums[y]), x, y])

            else:
                continue
        ac.st(len(ans))
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def abc_237d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc237/tasks/abc237_d
        tag: double_linked_list|implemention|classical
        """
        n = ac.read_int()
        pre = [-1] * (n + 3)
        post = [-1] * (n + 3)
        post[n + 1] = 0
        pre[0] = n + 1
        post[0] = n + 2
        pre[n + 2] = 0
        s = ac.read_str()
        for i in range(1, n + 1):
            if s[i - 1] == "L":
                pre[i] = pre[i - 1]
                post[pre[i - 1]] = i
                pre[i - 1] = i
                post[i] = i - 1
            else:
                post[i] = post[i - 1]
                pre[post[i - 1]] = i
                post[i - 1] = i
                pre[i] = i - 1
        ans = [n + 1]
        while ans[-1] != n + 2:
            ans.append(post[ans[-1]])
        ac.lst(ans[1:-1])
        return

    @staticmethod
    def cf_1922d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1922/D
        tag: double_linked_list|bfs|classical|brain_teaser|implemention
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            n += 1
            a = [0] + ac.read_list_ints()
            d = [math.inf] + ac.read_list_ints()
            pre = list(range(-1, n - 1))
            post = list(range(1, n + 1))
            pre[0] = post[n - 1] = 0
            ans = []
            f = [0] * n
            stack = set(range(1, n))
            for _ in range(n - 1):
                y = set()
                for x in stack:
                    if a[pre[x]] + a[post[x]] > d[x]:
                        y.add(x)
                        f[x] = 1
                ans.append(len(y))
                stack = set()
                for u in y:
                    pre[post[u]] = pre[u]
                    post[pre[u]] = post[u]
                    if not f[pre[u]]:
                        stack.add(pre[u])
                    if not f[post[u]]:
                        stack.add(post[u])
            ac.lst(ans)
        return
