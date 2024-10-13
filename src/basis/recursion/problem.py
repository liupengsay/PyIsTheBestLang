"""
Algorithm：divide_and_conquer|recursion|n-tree|pre_order|mid_order|post_order|iteration
Description：recursion|iteration

====================================LeetCode====================================
1545（https://leetcode.cn/problems/find-kth-bit-in-nth-binary-string/）recursion|implemention
894（https://leetcode.cn/problems/all-possible-full-binary-trees/）catalan_num|recursion|implemention
880（https://leetcode.cn/problems/decoded-string-at-index/）recursion|implemention
932（https://leetcode.cn/problems/beautiful-array/description/）recursion|divide_and_conquer|construction
889（https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/）recursion|divide_and_conquer|construction
1028（https://leetcode.cn/problems/recover-a-tree-from-preorder-traversal/description/）pre_order|recursion|construction|2-tree
100447（https://leetcode.com/problems/find-the-k-th-character-in-string-game-ii/）recursion|implemention|data_range|reverse_order

=====================================LuoGu======================================
P1911（https://www.luogu.com.cn/problem/P1911）4-tree|recursion|matrix
P5461（https://www.luogu.com.cn/problem/P5461）recursion|4-tree|matrix
P5551（https://www.luogu.com.cn/problem/P5551）pre_order|2-tree|recursion
P5626（https://www.luogu.com.cn/problem/P5626）divide_and_conquer|dp|merge_sort
P2907（https://www.luogu.com.cn/problem/P2907）recursion|implemention
P7673（https://www.luogu.com.cn/problem/P7673）mid_order|recursion|2-tree
P1228（https://www.luogu.com.cn/problem/P1228）4-tree|divide_and_conquer|recursion|matrix
P1185（https://www.luogu.com.cn/problem/P1185）2-tree|recursion
P2101（https://www.luogu.com.cn/problem/P2101）divide_and_conquer|greedy|classical
P5551（https://www.luogu.com.cn/problem/P5551）recursion

===================================CodeForces===================================
448C（https://codeforces.com/contest/448/problem/C）greedy|recursion|dp
1811D（https://codeforces.com/contest/1811/problem/D）recursion|fibonacci
559B（https://codeforces.com/problemset/problem/559/B）divide_and_conquer|implemention|string_hash
1400E（https://codeforces.com/problemset/problem/1400/E）divide_and_conquer|greedy|classical

===================================AcWing===================================
98（https://www.acwing.com/problem/content/100/）4-tree|recursion|matrix_rotate
93（https://www.acwing.com/problem/content/95/）recursion|comb|iteration
118（https://www.acwing.com/problem/content/120/）recursion

===================================AtCoder===================================
ABC350F（https://atcoder.jp/contests/abc350/tasks/abc350_f）implemention|divide_and_conquer|recursion|classical

"""
from functools import lru_cache
from itertools import combinations
from typing import Optional, List

from src.basis.tree_node.template import TreeNode
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_880(s: str, k: int) -> str:
        """
        url: https://leetcode.cn/problems/decoded-string-at-index/
        tag: recursion|implemention|iteration|classical
        """
        ans = ""
        while not ans:
            n = len(s)
            cur = 0
            for i in range(n):
                if s[i].isnumeric():
                    d = int(s[i])
                    if cur * d >= k:
                        s, k = s[:i], k % cur + cur * int(k % cur == 0)
                        break
                    cur *= d
                else:
                    if cur + 1 == k:
                        ans = s[i]
                        break
                    cur += 1
        return ans

    @staticmethod
    def lc_889(preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """
        url: https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/
        tag: recursion|divide_and_conquer|construction
        """
        tree = dict()
        m, n = len(preorder), len(postorder)
        stack = [(0, m - 1, 0, n - 1, 1)]
        ind = {val: i for i, val in enumerate(postorder)}
        while stack:
            i1, j1, i2, j2, i = stack.pop()
            if i >= 0:
                if i1 > j1:
                    continue
                root = preorder[i1]
                tree[i] = TreeNode(root)
                if i1 == j1:
                    continue
                stack.append((i1, j1, i2, j2, ~i))
                val = preorder[i1 + 1]
                x = ind[val]
                left_cnt = x - i2 + 1
                stack.append((i1 + 1, left_cnt + i1, i2, left_cnt + i2 - 1, i << 1))
                stack.append((left_cnt + i1 + 1, j1, left_cnt + i2, j2, (i << 1) | 1))
            else:
                i = ~i
                tree[i].left = tree.get(i << 1, None)
                tree[i].right = tree.get((i << 1) | 1, None)
        return tree[1]

    @lru_cache(None)
    def lc_894(self, n: int) -> List[Optional[TreeNode]]:
        """
        url: https://leetcode.cn/problems/all-possible-full-binary-trees/
        tag: catalan_number|recursion|implemention|classical|iteration
        """
        dp = [[] for _ in range(21)]
        dp[0] = []
        dp[1] = [TreeNode(0)]
        for i in range(2, 21):
            if i % 2 == 0:
                continue
            for j in range(1, i - 1):
                for left in dp[j]:
                    for right in dp[i - 1 - j]:
                        node = TreeNode(0)
                        node.left = left
                        node.right = right
                        dp[i].append(node)
        return dp[n]

    @staticmethod
    def lc_932(n: int) -> List[int]:
        """
        url: https://leetcode.cn/problems/beautiful-array/description/
        tag: recursion|divide_and_conquer|construction
        """
        m = 1000
        dp = [[], [1]]
        for i in range(2, m + 1):
            left = (i + 1) // 2
            right = i - left
            dp.append([2 * x - 1 for x in dp[left]] + [2 * x for x in dp[right]])
        return dp[n]

    @staticmethod
    def lc_1028(traversal: str) -> Optional[TreeNode]:
        """
        url: https://leetcode.cn/problems/recover-a-tree-from-preorder-traversal/description/
        tag: pre_order|recursion|construction|2-tree
        """
        tree = dict()
        stack = [(traversal, 1, 1)]
        while stack:
            s, i, d = stack.pop()
            if i >= 0:
                if s.isnumeric():
                    tree[i] = TreeNode(int(s))
                    continue
                stack.append(("", ~i, d))
                sp = "-" * d
                m = len(s)
                lst = []
                for j in range(m - d - 1):
                    if s[j].isnumeric() and s[j + d + 1].isnumeric() and s[j + 1:j + d + 1] == sp:
                        lst.append(j + 1)
                tree[i] = TreeNode(int(s[:lst[0]]))
                if len(lst) == 1:
                    stack.append((s[lst[0] + d:], i << 1, d + 1))
                else:
                    stack.append((s[lst[0] + d:lst[1]], i << 1, d + 1))
                    stack.append((s[lst[1] + d:], (i << 1) | 1, d + 1))
            else:
                i = ~i
                tree[i].left = tree.get(i << 1, None)
                tree[i].right = tree.get((i << 1) | 1, None)
        return tree[1]

    @staticmethod
    def lc_1545(n: int, k: int) -> str:
        """
        url: https://leetcode.cn/problems/find-kth-bit-in-nth-binary-string/
        tag: recursion|implemention
        """
        ans = ""
        flag = 0
        while not ans:
            if n == 1:
                ans = "0"[k - 1]
            elif n == 2:
                ans = "011"[k - 1]
            else:
                length = (1 << n) - 1
                if k == (length + 1) // 2:
                    ans = "1"
                elif k < (length + 1) // 2:
                    n -= 1
                else:
                    flag = 1 - flag
                    n -= 1
                    k = length // 2 - (k - length // 2) + 1
        return ans if not flag else str(1 - int(ans))

    @staticmethod
    def lg_p1911(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1911
        tag: 4-tree|recursion|matrix
        """
        n, x, y = ac.read_list_ints()
        ans = [[-1] * (1 << n) for _ in range(1 << n)]
        x -= 1
        y -= 1
        ans[x][y] = 0
        ind = 1
        stack = [[0, 0, (1 << n) - 1, (1 << n) - 1, x, y]]
        while stack:
            x1, y1, x2, y2, x, y = stack.pop()
            if x1 == x2 - 1:
                for i in range(x1, x2 + 1):
                    for j in range(y1, y2 + 1):
                        if ans[i][j] == -1:
                            ans[i][j] = ind
                ind += 1
                continue
            x0 = x1 + (x2 - x1) // 2
            y0 = y1 + (y2 - y1) // 2
            rec = [[x1, y1, x0, y0], [x1, y0 + 1, x0, y2], [x0 + 1, y1, x2, y0], [x0 + 1, y0 + 1, x2, y2]]
            center = [[x0, y0], [x0, y0 + 1], [x0 + 1, y0], [x0 + 1, y0 + 1]]
            for i in range(4):
                if rec[i][0] <= x <= rec[i][2] and rec[i][1] <= y <= rec[i][3]:
                    stack.append(rec[i] + [x, y])
                else:
                    ans[center[i][0]][center[i][1]] = ind
                    stack.append(rec[i] + center[i])
            ind += 1
        dct = dict()
        ind = 1
        for i in range(1 << n):
            for j in range(1 << n):
                x = ans[i][j]
                if not x:
                    continue
                if x not in dct:
                    dct[x] = ind
                    ind += 1
                ans[i][j] = dct[x]
        for ls in ans:
            ac.lst(ls)
        return

    @staticmethod
    def cf_448c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/448/problem/C
        tag: greedy|recursion|dp|implemention
        """
        ac.read_int()
        nums = ac.read_list_ints()

        @ac.bootstrap
        def dfs(arr):
            m = len(arr)
            low = min(arr)
            cur = [num - low for num in arr]
            ans = low
            i = 0
            while i < m:
                if cur[i] == 0:
                    i += 1
                    continue
                j = i
                while j < m and cur[j] > 0:
                    j += 1
                ans += yield dfs(cur[i: j])
                i = j
            yield ac.min(ans, m)

        ac.st(dfs(nums))
        return

    @staticmethod
    def ac_98(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/100/
        tag: 4-tree|recursion|matrix_rotate
        """
        for _ in range(ac.read_int()):
            n, a, b = ac.read_list_ints()
            a -= 1
            b -= 1

            def check(nn, mm):
                stack = [[nn, mm]]
                x = y = -1
                while stack:
                    if stack[-1][0] == 0:
                        x = y = 0
                        stack.pop()
                        continue
                    else:
                        nn, mm = stack[-1]
                        cc = 2 ** (2 * nn - 2)
                        if x != -1:
                            stack.pop()
                            z = mm // cc
                            length = 2 ** (nn - 1)
                            if z == 0:
                                x, y = y, x
                            elif z == 1:
                                x, y = x, y + length
                            elif z == 2:
                                x, y = x + length, y + length
                            else:
                                x, y = 2 * length - y - 1, length - x - 1
                        else:
                            stack.append([nn - 1, mm % cc])
                return x, y

            x1, y1 = check(n, a)
            x2, y2 = check(n, b)
            ans = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 * 10
            ans = int(ans) + int(ans - int(ans) >= 0.5)
            ac.st(ans)
        return

    @staticmethod
    def ac_93_1(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/95/
        tag: recursion|comb|iteration|back_trace
        """
        n, m = ac.read_list_ints()

        def dfs(i):
            if len(pre) == m:
                ac.lst(pre)
                return
            if i == n:
                return

            dfs(i + 1)
            pre.append(i + 1)
            dfs(i + 1)
            pre.pop()
            return

        pre = []
        dfs(0)
        return

    @staticmethod
    def ac_93_2(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/95/
        tag: recursion|comb|iteration
        """
        n, m = ac.read_list_ints()

        pre = []
        stack = [[0, 0]]
        while stack:
            i, state = stack.pop()
            if i >= 0:
                stack.append([~i, state])
                if len(pre) == m:
                    ac.lst(pre)
                    continue
                if i == n:
                    continue
                stack.append([i + 1, 0])
                pre.append(i + 1)
                stack.append([i + 1, 1])
            else:
                if state:
                    pre.pop()
        return

    @staticmethod
    def ac_93_3(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/95/
        tag: recursion|comb|iteration
        """
        n, m = ac.read_list_ints()
        for item in combinations(list(range(1, n + 1)), m):
            ac.lst(list(item))
        return

    @staticmethod
    def ac_118(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/120/
        tag: recursion
        """
        dp = [[["X"]]]
        for m in range(2, 8):
            n = 3 ** (m - 1)
            cur = [[" "] * n for _ in range(n)]
            k = 3 ** (m - 2)
            start = [[0, 0], [0, 2 * k], [k, k], [2 * k, 0], [2 * k, 2 * k]]
            for x, y in start:
                for i in range(x, x + k):
                    for j in range(y, y + k):
                        cur[i][j] = dp[-1][i - x][j - y]
            dp.append([ls[:] for ls in cur])
        while True:
            x = ac.read_int()
            if x == -1:
                break
            for ls in dp[x - 1]:
                ac.st("".join(ls))
            ac.st("-")
        return

    @staticmethod
    def abc_350f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc350/tasks/abc350_f
        tag: implemention|divide_and_conquer|recursion|classical
        """
        s = ac.read_str()
        n = len(s)
        right = [-1] * n
        stack = []
        for i in range(n):
            if s[i] == "(":
                stack.append(i)
            elif s[i] == ")":
                right[stack.pop()] = i

        pre = ac.accumulate([int(w not in "()") for w in s])
        ans = [""] * pre[-1]
        dct = dict()
        for i in range(26):
            dct[chr(ord("a") + i)] = chr(ord("A") + i)
            dct[chr(ord("A") + i)] = chr(ord("a") + i)
        stack = [(0, n - 1, 0, pre[-1] - 1, 0)]
        while stack:
            a, b, ll, rr, state = stack.pop()
            if right[a] == -1:
                if not state:
                    ans[ll] = s[a]
                    ll += 1
                else:
                    ans[rr] = dct[s[a]]
                    rr -= 1
                if a + 1 <= b and pre[b + 1] - pre[a + 1]:
                    stack.append((a + 1, b, ll, rr, state))
            else:
                bb = right[a]
                if a + 1 <= bb - 1:
                    cnt = pre[bb] - pre[a + 1]
                    if cnt:
                        if not state:
                            stack.append((a + 1, bb - 1, ll, ll + cnt - 1, state ^ 1))
                        else:
                            stack.append((a + 1, bb - 1, rr - cnt + 1, rr, state ^ 1))

                if bb + 1 <= b:
                    cnt = pre[b + 1] - pre[bb + 1]
                    if cnt:
                        if not state:
                            stack.append((bb + 1, b, rr - cnt + 1, rr, state))
                        else:
                            stack.append((bb + 1, b, ll, ll + cnt - 1, state))
        ac.st("".join(ans))
        return

    @staticmethod
    def cf_559b(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/559/B
        tag: divide_and_conquer|implemention|string_hash
        """

        a = ac.read_str()
        b = ac.read_str()

        def check(s):
            def dfs(i, j):
                if (j - i + 1) % 2:
                    return s[i:j + 1]

                mid = i + (j - i + 1) // 2 - 1
                s1 = dfs(i, mid)
                s2 = dfs(mid + 1, j)
                return s1 + s2 if s1 < s2 else s2 + s1

            return dfs(0, len(s) - 1)

        ans = check(a) == check(b)
        ac.st("YES" if ans else "NO")
        return

    @staticmethod
    def lg_p2101(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2101
        tag: divide_and_conquer|greedy|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()

        def dfs(i, j):
            cur = j-i+1
            low = i
            for k in range(i, j + 1):
                if nums[k] < nums[low]:
                    low = k
            res = nums[low]
            for k in range(i, j + 1):
                nums[k] -= res
            cnt = 0
            for k in range(i, j + 1):
                if nums[k]:
                    cnt += 1
                else:
                    if cnt:
                        res += dfs(k - cnt, k - 1)
                    cnt = 0
            if cnt:
                res += dfs(j - cnt + 1, j)
            return min(res, cur)

        ans = dfs(0, n - 1)
        ac.st(ans)
        return

    @staticmethod
    def cf_1400e(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1400/E
        tag: divide_and_conquer|greedy|classical
        """
        n = ac.read_int()
        nums = ac.read_list_ints()  # MLE
        dp = [math.inf] * n * n

        sub = [[] for _ in range(n * n)]
        stack = [n - 1]
        while stack:
            val = stack.pop()
            if val >= 0:
                stack.append(~val)
                i, j = val // n, val % n
                low = i
                for k in range(i, j + 1):
                    if nums[k] < nums[low]:
                        low = k
                res = nums[low]
                for k in range(i, j + 1):
                    nums[k] -= res
                dp[val] = res

                cnt = 0
                for k in range(i, j + 1):
                    if nums[k]:
                        cnt += 1
                    else:
                        if cnt:
                            stack.append((k - cnt) * n + k - 1)
                            sub[val].append((k - cnt) * n + k - 1)
                        cnt = 0
                if cnt:
                    stack.append((j - cnt + 1) * n + j)
                    sub[val].append((j - cnt + 1) * n + j)
            else:
                val = ~val
                i, j = val // n, val % n
                for y in sub[val]:
                    dp[val] += dp[y]
                dp[val] = min(dp[val], j - i + 1)
        ans = dp[n - 1]
        ac.st(ans)
        return
