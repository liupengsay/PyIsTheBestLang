import unittest

from typing import List
from collections import defaultdict, Counter

import random

from src.basis.binary_search import BinarySearch
from src.fast_io import FastIO

"""

算法：字符串哈希、树哈希、矩阵哈希、树的最小表示法
功能：将一定长度的字符串映射为多项式函数值，并进行比较或者计数，通常结合滑动窗口进行计算，注意防止哈希碰撞
题目：

===================================力扣===================================
214. 最短回文串（https://leetcode.cn/problems/shortest-palindrome/）使用正向与反向字符串哈希计算字符串前缀最长回文子串
572. 另一棵树的子树（https://leetcode.cn/problems/subtree-of-another-tree/）经典树结构哈希
1044. 最长重复子串（https://leetcode.cn/problems/shortest-palindrome/）利用二分查找加字符串哈希确定具有最长长度的重复子串
1316. 不同的循环子字符串（https://leetcode.cn/problems/shortest-palindrome/）利用字符串哈希确定不同循环子串的个数
2156 查找给定哈希值的子串（https://leetcode.cn/problems/find-substring-with-given-hash-value/）逆向进行字符串哈希的计算
652. 寻找重复的子树（https://leetcode.cn/problems/find-duplicate-subtrees/）树哈希，确定重复子树
1554. 只有一个不同字符的字符串（https://leetcode.cn/problems/strings-differ-by-one-character/）字符串前后缀哈希求解
1923. 最长公共子路径（https://leetcode.cn/problems/longest-common-subpath/）经典二分查找加滚动哈希
1948. 删除系统中的重复文件夹（https://leetcode.cn/problems/delete-duplicate-folders-in-system/）字典树与树哈希去重

===================================洛谷===================================
P8835 [传智杯 #3 决赛] 子串（https://www.luogu.com.cn/record/list?user=739032&status=12&page=14）字符串哈希或者KMP查找匹配的连续子串
P6140 [USACO07NOV]Best Cow Line S（https://www.luogu.com.cn/problem/P6140）贪心模拟与字典序比较，使用字符串哈希与二分查找比较正序与倒序最长公共子串
P2870 [USACO07DEC]Best Cow Line G（https://www.luogu.com.cn/problem/P2870）贪心模拟与字典序比较，使用字符串哈希与二分查找比较正序与倒序最长公共子串
P5832 [USACO19DEC]Where Am I? B（https://www.luogu.com.cn/problem/P5832）可以使用字符串哈希进行最长的长度使得所有对应长度的子串均是唯一的
P2852 [USACO06DEC]Milk Patterns G（https://www.luogu.com.cn/problem/P2852）二分加字符串哈希计算出现超过 k 次的最长连续子数组
P4656 [CEOI2017] Palindromic Partitions（https://www.luogu.com.cn/problem/P4656）使用字符串哈希贪心选取
P6739 [BalticOI 2014 Day1] Three Friends（https://www.luogu.com.cn/problem/P6739）前后缀字符串哈希

================================CodeForces================================
D. Remove Two Letters（https://codeforces.com/problemset/problem/1800/D）字符串前后缀哈希加和变换

================================AcWing================================
138. 兔子与兔子（https://www.acwing.com/problem/content/140/）字符串哈希，计算子串是否完全相等
156. 矩阵（https://www.acwing.com/problem/content/description/158/）经典矩阵哈希
157. 树形地铁系统（https://www.acwing.com/problem/content/description/159/）经典树哈希，树的最小表示法

参考：OI WiKi（xx）
"""


class StringHash:
    # 注意哈希碰撞，需要取两个质数与模进行区分
    def __init__(self, n, s):
        self.n = n
        self.p = [random.randint(26, 100), random.randint(26, 100)]
        self.mod = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]
        self.pre = [[0], [0]]
        self.pp = [[1], [1]]
        for w in s:
            for i in range(2):
                self.pre[i].append((self.pre[i][-1] * self.p[i] + ord(w) - ord("a")) % self.mod[i])
                self.pp[i].append((self.pp[i][-1] * self.p[i]) % self.mod[i])
        return

    def query(self, x, y):
        # 模板：字符串区间的哈希值，索引从 0 开始
        ans = [0, 0]
        for i in range(2):
            if x <= y:
                ans[i] = (self.pre[i][y + 1] - self.pre[i][x] * self.pp[i][y-x+1]) % self.mod[i]
        return ans


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1800g(ac=FastIO()):
        # 模板：使用树哈希编码判断树是否对称
        for _ in range(ac.read_int()):
            n = ac.read_int()
            edge = [[] for _ in range(n)]
            for _ in range(n-1):
                u, v = ac.read_ints_minus_one()
                edge[u].append(v)
                edge[v].append(u)

            @ac.bootstrap
            def dfs(i, fa):
                res = []
                cnt = 1
                for j in edge[i]:
                    if j != fa:
                        yield dfs(j, i)
                        res.append(tree_hash[j])
                        cnt += sub[j]
                st = (tuple(sorted(res)), cnt)
                if st not in seen:
                    seen[st] = len(seen) + 1
                tree_hash[i] = seen[st]
                sub[i] = cnt
                yield

            # 使用深搜或者迭代预先将子树进行哈希编码
            tree_hash = [-1] * n
            sub = [0]*n
            seen = dict()
            dfs(0, -1)

            # 逐层判断哈希值不为0的子树是否对称
            u = 0
            father = -1
            ans = "YES"
            while u != -1:
                dct = Counter(tree_hash[v] for v in edge[u] if v != father)
                single = 0
                for w in dct:
                    single += dct[w] % 2
                if single == 0:
                    break
                if single > 1:
                    ans = "NO"
                    break
                for v in edge[u]:
                    if v != father and dct[tree_hash[v]] % 2:
                        u, father = v, u
                        break
            ac.st(ans)
        return

    @staticmethod
    def lc_652(root):
        # 使用树哈希编码序列化子树，查找重复子树
        def dfs(node):
            if not node:
                return 0

            state = (node.val, dfs(node.left), dfs(node.right))
            if state in seen:
                node, idy = seen[state]
                repeat.add(node)
                return idy
            seen[state] = [node, len(seen) + 1]
            return seen[state][1]

        seen = dict()
        repeat = set()
        dfs(root)
        return list(repeat)

    @staticmethod
    def cf_1800d(ac=FastIO()):

        # 模板：字符串前后缀哈希加和，使用两个哈希避免碰撞
        n = 2 * 10 ** 5
        p1 = random.randint(26, 100)
        p2 = random.randint(26, 100)
        mod1 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)
        mod2 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        dp1 = [1]
        for _ in range(1, n + 1):
            dp1.append((dp1[-1] * p1) % mod1)
        dp2 = [1]
        for _ in range(1, n + 1):
            dp2.append((dp2[-1] * p2) % mod2)

        for _ in range(ac.read_int()):
            n = ac.read_int()
            s = ac.read_str()

            post1 = [0] * (n + 1)
            for i in range(n - 1, -1, -1):
                post1[i] = (post1[i + 1] + (ord(s[i]) - ord("a")) * dp1[n - 1 - i]) % mod1

            post2 = [0] * (n + 1)
            for i in range(n - 1, -1, -1):
                post2[i] = (post2[i + 1] + (ord(s[i]) - ord("a")) * dp2[n - 1 - i]) % mod2

            ans = set()
            pre1 = pre2 = 0
            for i in range(n - 1):
                x1 = pre1
                y1 = post1[i + 2]
                x2 = pre2
                y2 = post2[i + 2]
                ans.add(((x1 * dp1[n - i - 2] + y1) % mod1, (x2 * dp2[n - i - 2] + y2) % mod2))
                pre1 = (pre1 * p1) % mod1 + ord(s[i]) - ord("a")
                pre2 = (pre2 * p2) % mod2 + ord(s[i]) - ord("a")
            ac.st(len(ans))
        return

    @staticmethod
    def ac_138(ac=FastIO()):

        # 模板：字符串哈希，计算子串是否完全相等
        p1 = random.randint(26, 100)
        p2 = random.randint(26, 100)
        mod1 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)
        mod2 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        s = ac.read_str()
        n = len(s)
        pre1 = [0] * (n + 1)
        pre2 = [0] * (n + 1)
        for i in range(n):
            pre1[i + 1] = (pre1[i] * p1 + ord(s[i]) - ord("a")) % mod1
            pre2[i + 1] = (pre2[i] * p2 + ord(s[i]) - ord("a")) % mod2
        m = ac.read_int()
        lst = []
        while len(lst) < m * 4:
            lst.extend(ac.read_list_ints_minus_one())

        for i in range(0, m * 4, 4):
            x1, y1, x2, y2 = lst[i: i + 4]
            m1 = y1 - x1 + 1
            m2 = y2 - x2 + 1
            if m1 != m2:
                ac.st("No")
                continue
            cur1 = ((pre1[y1 + 1] - pre1[x1] * pow(p1, m1, mod1)) % mod1,
                    (pre2[y1 + 1] - pre2[x1] * pow(p2, m2, mod2)) % mod2)
            cur2 = ((pre1[y2 + 1] - pre1[x2] * pow(p1, m1, mod1)) % mod1,
                    (pre2[y2 + 1] - pre2[x2] * pow(p2, m2, mod2)) % mod2)
            if cur1 == cur2:
                ac.st("Yes")
            else:
                ac.st("No")
        return

    @staticmethod
    def ac_156(ac=FastIO()):
        # 模板：二维矩阵哈希查找子矩阵是否存在
        m, n, a, b = ac.read_ints()
        grid = [ac.read_str() for _ in range(m)]

        # 经典双哈希防止碰撞
        p1 = random.randint(26, 100)
        p2 = random.randint(26, 100)
        mod1 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)
        mod2 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        def check(p, mod):

            # 先列从左到右，按行从上到下
            col = [[0] * n for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    col[i + 1][j] = (col[i][j] * p + int(grid[i][j])) % mod

            # 计算当前点往列向上 a 长度的哈希值
            pa = pow(p % mod, a, mod)
            for j in range(n):
                for i in range(m - 1, a - 2, -1):
                    col[i + 1][j] = (col[i + 1][j] - col[i - a + 1][j] * pa) % mod

            # 计算每一个形状为 a*b 的子矩阵哈希值
            pre = set()
            pab = pow(pa, b, mod)  # 注意此时的模数
            for i in range(a - 1, m):
                lst = [0]
                x = 0
                for j in range(n):
                    x *= pa
                    x += col[i + 1][j]
                    x %= mod
                    lst.append(x)
                    if j >= b - 1:
                        # 计算向左 b 长度的哈希值
                        cur = (lst[j + 1] - (lst[j - b + 1] % mod) * pab) % mod
                        pre.add(cur)
            return pre

        def check_read(p, mod):
            # 先列从左到右，按行从上到下
            y = 0
            for j in range(b):
                for i in range(a):
                    y *= p
                    y += int(grid[i][j])
                    y %= mod
            # 形式为 [[p^3, p^1], [p^2, p^0]]
            return y

        pre1 = check(p1, mod1)
        pre2 = check(p2, mod2)

        for _ in range(ac.read_int()):
            grid = [ac.read_str() for _ in range(a)]
            y1 = check_read(p1, mod1)
            y2 = check_read(p2, mod2)
            if y1 in pre1 and y2 in pre2:
                ac.st(1)
            else:
                ac.st(0)
        return

    @staticmethod
    def ac_157(ac=FastIO()):

        def check(st):
            # 模板：解码原始树的字符串表示，再计算树的最小表示法

            parent = [-1]
            pa = 0
            ind = 0
            dct = defaultdict(list)
            for w in st:
                if w == "0":
                    ind += 1
                    dct[pa].append(ind)
                    parent.append(pa)
                    pa = ind
                else:
                    pa = parent[pa]

            # 生成树的最小表示法
            n = ind + 1
            stack = [0]
            sub = [""]*n
            while stack:
                i = stack.pop()
                if i >= 0:
                    stack.append(~i)
                    for j in dct[i]:
                        stack.append(j)
                else:
                    i = ~i
                    lst = []
                    for j in dct[i]:
                        lst.append("0"+sub[j]+"1")
                        sub[j] = ""
                    lst.sort()
                    sub[i] = "".join(lst)
            return sub[0]

        for _ in range(ac.read_int()):
            s = ac.read_str()
            t = ac.read_str()
            if check(s) == check(t):
                ac.st("same")
            else:
                ac.st("different")
        return

    @staticmethod
    def lg_p2852(ac=FastIO()):

        # 模板；二分加字符串哈希计算出现超过 k 次的最长连续子数组
        p1 = random.randint(26, 100)
        p2 = random.randint(26, 100)
        mod1 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)
        mod2 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        def check(x):
            pre = defaultdict(int)
            s1 = s2 = 0
            pow1 = pow(p1, x - 1, mod1)
            pow2 = pow(p2, x - 1, mod2)
            for i in range(n):
                s1 = (s1 * p1 + nums[i]) % mod1
                s2 = (s2 * p2 + nums[i]) % mod2
                if i >= x - 1:
                    pre[(s1, s2)] += 1
                    s1 = (s1 - pow1 * nums[i - x + 1]) % mod1
                    s2 = (s2 - pow2 * nums[i - x + 1]) % mod2
            return max(pre.values()) >= k

        n, k = ac.read_ints()
        nums = [ac.read_int() for _ in range(n)]
        ans = BinarySearch().find_int_right(0, n, check)
        ac.st(ans)
        return

    @staticmethod
    def lg_p4656(ac=FastIO()):
        # 模板：使用字符串哈希贪心选取

        p1 = random.randint(26, 100)
        p2 = random.randint(26, 100)
        mod1 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)
        mod2 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        for _ in range(ac.read_int()):
            s = ac.read_str()
            ans = 0
            n = len(s)
            i, j = 0, n - 1
            while j - i + 1 >= 2:
                # 从两边依次进行选取
                flag = False
                pre1 = post1 = pre2 = post2 = 0
                pp1 = pp2 = 1
                x, y = i, j
                while True:
                    if pre1 == post1 and pre2 == post2 and x > i:
                        flag = True
                        i = x
                        j = y
                        break
                    if y - x + 1 <= 1:
                        break
                    w = s[x]
                    pre1 = (pre1 * p1 + ord(w) - ord("a")) % mod1
                    pre2 = (pre2 * p2 + ord(w) - ord("a")) % mod2

                    w = s[y]
                    post1 = (post1 + pp1 * (ord(w) - ord("a"))) % mod1
                    post2 = (post2 + pp2 * (ord(w) - ord("a"))) % mod2
                    pp1 = (pp1 * p1) % mod1
                    pp2 = (pp2 * p2) % mod2
                    x += 1
                    y -= 1
                # 如果构成一对回文增加 2 否则增加 1
                if flag:
                    ans += 2
                else:
                    ans += 1
                    i = j + 1
                    break
            # 特判还剩中间一个字母的情况
            if i == j:
                ans += 1
            ac.st(ans)

        return

    @staticmethod
    def lg_p6739(ac=FastIO()):
        # 模板：前后缀字符串哈希
        n = ac.read_int()
        s = ac.read_str()
        sth = StringHash(n, s)

        ans = dict()
        for i in range(n):
            if len(ans) > 1:
                break
            if i < n // 2:
                ss = sth.query(0, i - 1)
                tt = sth.query(i + 1, n // 2)
                pp = [(ss[j] * sth.pp[j][n // 2 - i] + tt[j]) % sth.mod[j] for j in range(2)]

                rr = sth.query(n // 2 + 1, n - 1)
                if pp == rr:
                    ans[tuple(pp)] = i

            elif i == n // 2:
                pp = sth.query(0, n // 2 - 1)
                rr = sth.query(n // 2 + 1, n - 1)
                if pp == rr:
                    ans[tuple(pp)] = i
            else:
                pp = sth.query(0, n // 2 - 1)

                ss = sth.query(n // 2, i - 1)
                tt = sth.query(i + 1, n - 1)
                rr = [(ss[j] * sth.pp[j][n - 1 - i] + tt[j]) % sth.mod[j] for j in range(2)]
                if pp == rr:
                    ans[tuple(pp)] = i
        if not ans:
            ac.st("NOT POSSIBLE")
        elif len(ans) > 1:
            ac.st("NOT UNIQUE")
        else:
            i = list(ans.values())[0]
            ac.st((s[:i] + s[i + 1:])[:n // 2])
        return

    @staticmethod
    def lc_1554(lst: List[str]) -> bool:
        # 模板：字符串前后缀哈希求解
        m = len(lst[0])
        p = [random.randint(26, 100), random.randint(26, 100)]
        mod = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]

        pre_hash = []
        post_hash = []
        for s in lst:
            pre = [[0], [0]]
            pp = [[1], [1]]
            for w in s:
                for i in range(2):
                    pre[i].append((pre[i][-1] * p[i] + ord(w) - ord("a")) % mod[i])
                    pp[i].append((pp[i][-1] * p[i]) % mod[i])
            pre_hash.append(pre[:])

            pre = [[0], [0]]
            pp = [[1], [1]]
            for w in s[::-1]:
                for i in range(2):
                    pre[i].append((pre[i][-1] * p[i] + ord(w) - ord("a")) % mod[i])
                    pp[i].append((pp[i][-1] * p[i]) % mod[i])
            post_hash.append([p[::-1] for p in pre])

        n = len(lst)
        for i in range(m):
            pre = set()
            for j in range(n):
                va = tuple()
                for k in range(2):
                    va += tuple([pre_hash[j][k][i]]) + tuple([post_hash[j][k][i+1]])
                if va in pre:
                    return True
                pre.add(va)
        return False

    @staticmethod
    def lc_1316(text: str) -> int:
        # 模板：字符串哈希判断循环子串
        n = len(text)
        sh = StringHash(n, text)

        ans = 0
        for x in range(1, n // 2 + 1):
            cur = set()
            for i in range(n - 2 * x + 1):
                ans1 = sh.query(i, i + x - 1)
                ans2 = sh.query(i + x, i + 2 * x - 1)
                if ans1 == ans2:
                    # 注意只有长度与哈希值相同字符串才相同
                    cur.add(tuple(ans1))
            ans += len(cur)
        return ans


class TestGeneral(unittest.TestCase):

    def test_string_hash(self):
        n = 1000
        st = "".join([chr(random.randint(0, 25) + ord("a")) for _ in range(n)])

        # 生成哈希种子
        p1, p2 = random.randint(26, 100), random.randint(26, 100)
        mod1, mod2 = random.randint(
            10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        # 计算目标串的哈希状态
        target = "".join([chr(random.randint(0, 25) + ord("a"))
                         for _ in range(10)])
        h1 = h2 = 0
        for w in target:
            h1 = h1 * p1 + (ord(w) - ord("a"))
            h1 %= mod1
            h2 = h2 * p2 + (ord(w) - ord("a"))
            h2 %= mod1

        # 滑动窗口计算哈希状态
        m = len(target)
        pow1 = pow(p1, m - 1, mod1)
        pow2 = pow(p2, m - 1, mod2)
        s1 = s2 = 0
        cnt = 0
        n = len(st)
        for i in range(n):
            w = st[i]
            s1 = s1 * p1 + (ord(w) - ord("a"))
            s1 %= mod1
            s2 = s2 * p2 + (ord(w) - ord("a"))
            s2 %= mod1
            if i >= m - 1:
                if (s1, s2) == (h1, h2):
                    cnt += 1
                s1 = s1 - (ord(st[i - m + 1]) - ord("a")) * pow1
                s1 %= mod1
                s2 = s2 - (ord(st[i - m + 1]) - ord("a")) * pow2
                s2 %= mod1
            if st[i:i + m] == target:
                cnt -= 1
        assert cnt == 0
        return


if __name__ == '__main__':
    unittest.main()
