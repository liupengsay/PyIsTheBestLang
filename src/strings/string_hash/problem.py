"""
Algorithm：string_hash|tree_hash|matrix_hash|tree_minimum_expression|longest_prefix_palindrome_substring|longest_suffix_palindrome_substring
Description：counter|sliding_window|double_random_mod|hash_crush

====================================LeetCode====================================
214（https://leetcode.cn/problems/shortest-palindrome/）reverse_hash|string_hash|longest_prefix_palindrome_substring|kmp|manacher
572（https://leetcode.cn/problems/subtree-of-another-tree/）tree_hash
1044（https://leetcode.cn/problems/shortest-palindrome/）suffix_array|height|classical|string_hash
1316（https://leetcode.cn/problems/distinct-echo-substrings）string_hash
2156（https://leetcode.cn/problems/find-substring-with-given-hash-value/）reverse_hash|string_hash
652（https://leetcode.cn/problems/find-duplicate-subtrees/）tree_hash
1554（https://leetcode.cn/problems/strings-differ-by-one-character/）string_hash|trie
1923（https://leetcode.cn/problems/longest-common-subpath/）binary_search|rolling_hash
1948（https://leetcode.cn/problems/delete-duplicate-folders-in-system/）trie_like|tree_hash
2261（https://leetcode.cn/problems/k-divisible-elements-subarrays/submissions/）string_hash
187（https://leetcode-cn.com/problems/repeated-dna-sequences/）
2851（https://leetcode.cn/problems/string-transformation/）string_hash|kmp|matrix_dp|matrix_fast_power
2977（https://leetcode.cn/problems/minimum-cost-to-convert-string-ii/）string_hash|dp|dijkstra|trie
100208（https://leetcode.com/contest/weekly-contest-385/problems/count-prefix-and-suffix-pairs-ii/）string_hash|brute_force
3327（https://leetcode.cn/problems/check-if-dfs-strings-are-palindromes/）dfs_order|manacher|palindrome|string_hash_single|classical

=====================================LuoGu======================================
P6140（https://www.luogu.com.cn/problem/P6140）greedy|implemention|lexicographical_order|string_hash|binary_search|reverse_order|lcs
P2870（https://www.luogu.com.cn/problem/P2870）greedy|implemention|lexicographical_order|string_hash|binary_search|reverse_order|lcs
P5832（https://www.luogu.com.cn/problem/P5832）string_hash
P2852（https://www.luogu.com.cn/problem/P2852）binary_search|suffix_array|height|monotonic_queue|string_hash
P4656（https://www.luogu.com.cn/problem/P4656）string_hash|greedy
P6739（https://www.luogu.com.cn/problem/P6739）prefix_suffix|string_hash
P3370（https://www.luogu.com.cn/problem/P3370）string_hash
P2601（https://www.luogu.com.cn/problem/P2601）matrix_hash
P4824（https://www.luogu.com.cn/problem/P4824）string_hash
P4503（https://www.luogu.com.cn/problem/P4503）string_hash
P3538（https://www.luogu.com.cn/problem/P3538）string_hash|prime_factor|brute_force|circular_section
P6312（https://www.luogu.com.cn/problem/P6312）string_hash|classical

===================================CodeForces===================================
1800D（https://codeforces.com/contest/1800/problem/D）prefix_suffix|hash
514C（https://codeforces.com/problemset/problem/514/C）string_hash
1200E（https://codeforces.com/problemset/problem/1200/E）string_hash|kmp
580E（https://codeforces.com/problemset/problem/580/E）segment_tree_hash|range_change|range_hash_reverse|circular_section
452F（https://codeforces.com/contest/452/problem/F）segment_tree_hash|string_hash|point_set|range_hash|range_reverse
7D（https://codeforces.com/problemset/problem/7/D）string_hash|palindrome|classical
835D（https://codeforces.com/problemset/problem/835/D）palindrome|string_hash
1977D（https://codeforces.com/contest/1977/problem/D）string_hash|brute_force|brain_teaser|classical
1418G（https://codeforces.com/problemset/problem/1418/G）string_hash|random_hash|classical|two_pointers

====================================AtCoder=====================================
ABC141E（https://atcoder.jp/contests/abc141/tasks/abc141_e）binary_search|string_hash|check
ABC331F（https://atcoder.jp/contests/abc331/tasks/abc331_f）point_set|range_hash_reverse|palindrome|classical
ABC310C（https://atcoder.jp/contests/abc310/tasks/abc310_c）string_hash|classical
ABC353E（https://atcoder.jp/contests/abc353/tasks/abc353_e）string_hash|trie
ABC367F（https://atcoder.jp/contests/abc367/tasks/abc367_f）random_seed|random_hash

=====================================AcWing=====================================
140（https://www.acwing.com/problem/content/140/）string_hash
158（https://www.acwing.com/problem/content/description/158/）matrix_hash
159（https://www.acwing.com/problem/content/description/159/）tree_hash|tree_minimum_expression
139（https://www.acwing.com/problem/content/139/）matrix_hash

=====================================LibraryChecker=====================================
1（https://ac.nowcoder.com/acm/contest/64384/D）string_hash|implemention
2（https://www.luogu.com.cn/problem/UVA11019）matrix_hash|string_hash
3（https://ac.nowcoder.com/acm/problem/51003）matrix_hash|string_hash

"""

import random
from collections import defaultdict, Counter
from itertools import accumulate
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.mathmatics.fast_power.template import MatrixFastPower
from src.mathmatics.prime_factor.template import PrimeFactor
from src.search.dfs.template import UnWeightedTree
from src.strings.string_hash.template import StringHash, PointSetRangeHashReverse, RangeSetRangeHashReverse, \
    MatrixHash, MatrixHashReverse, StringHashSingle, StringHashSingleBuild
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def cf_1800g(ac=FastIO()):
        # tree_hash编码判断树是否对称
        for _ in range(ac.read_int()):
            n = ac.read_int()
            edge = [[] for _ in range(n)]
            for _ in range(n - 1):
                u, v = ac.read_list_ints_minus_one()
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

            # dfs|或者迭代预先将子tree_hash编码
            tree_hash = [-1] * n
            sub = [0] * n
            seen = dict()
            dfs(0, -1)

            # 逐层判断hash值不为0的子树是否对称
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
    def lc_214(s: str) -> str:
        """
        url: https://leetcode.cn/problems/shortest-palindrome/
        tag: reverse_hash|string_hash|longest_prefix_palindrome_substring|kmp|manacher
        """

        # 正向与反向string_hash字符串前缀最长palindrome_substring，也可以用KMP与manacher

        def query(x, y):
            # 字符串区间的hash值，索引从 0 开始
            ans = [0, 0]
            for ii in range(2):
                if x <= y:
                    ans[ii] = (pre[ii][y + 1] - pre[ii][x] * pp[ii][y - x + 1]) % mod[ii]
            return ans

        def query_rev(x, y):
            # 字符串区间的hash值，索引从 0 开始
            ans = [0, 0]
            for ii in range(2):
                if x <= y:
                    ans[ii] = (rev[ii][y + 1] - rev[ii][x] * pp[ii][y - x + 1]) % mod[ii]
            return ans

        n = len(s)
        p = [random.randint(26, 100), random.randint(26, 100)]
        mod = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]
        pre = [[0], [0]]
        pp = [[1], [1]]
        for w in s:
            for i in range(2):
                pre[i].append((pre[i][-1] * p[i] + ord(w) - ord("a")) % mod[i])
                pp[i].append((pp[i][-1] * p[i]) % mod[i])

        rev = [[0], [0]]
        for w in s[::-1]:
            for i in range(2):
                rev[i].append((rev[i][-1] * p[i] + ord(w) - ord("a")) % mod[i])

        length = 1
        for i in range(1, n):
            m = (i + 1) // 2
            left = query(0, m - 1)
            right = query_rev((n - 1) - i, (n - 1) - (i - m + 1))
            if left == right:
                length = i + 1
        return s[length:][::-1] + s

    @staticmethod
    def lc_652(root):
        """
        url: https://leetcode.cn/problems/find-duplicate-subtrees/
        tag: tree_hash
        """

        # tree_hash编码序列化子树，查找重复子树
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
    def cf_1800d_1(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1800/problem/D
        tag: prefix_suffix|hash
        """

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
    def cf_1800d_2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1800/problem/D
        tag: prefix_suffix|hash
        """

        for _ in range(ac.read_int()):
            n = ac.read_int()
            s = ac.read_str()
            ans = n - 1
            for i in range(2, n):
                if s[i] == s[i - 2]:
                    ans -= 1
            ac.st(ans)
        return

    @staticmethod
    def abc_141e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc141/tasks/abc141_e
        tag: suffix_array|height|binary_search|string_hash
        """

        def check(x):
            if x == 0:
                return True
            pre = dict()
            for i in range(x - 1, n):
                cur = (sh1.query(i - x + 1, i), sh2.query(i - x + 1, i))
                if cur in pre:
                    if i - pre[cur] >= x:
                        return True
                else:
                    pre[cur] = i
            return False

        n = ac.read_int()
        s = ac.read_str()
        sh1 = StringHash([ord(w) - ord("a") for w in s])
        sh2 = StringHash([ord(w) - ord("a") for w in s])
        ans = BinarySearch().find_int_right(0, n, check)
        ac.st(ans)
        return

    @staticmethod
    def ac_138(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/140/
        tag: string_hash
        """
        # string_hash，子串是否完全相等
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
                ac.no()
                continue
            cur1 = ((pre1[y1 + 1] - pre1[x1] * pow(p1, m1, mod1)) % mod1,
                    (pre2[y1 + 1] - pre2[x1] * pow(p2, m2, mod2)) % mod2)
            cur2 = ((pre1[y2 + 1] - pre1[x2] * pow(p1, m1, mod1)) % mod1,
                    (pre2[y2 + 1] - pre2[x2] * pow(p2, m2, mod2)) % mod2)
            if cur1 == cur2:
                ac.yes()
            else:
                ac.no()
        return

    @staticmethod
    def ac_158(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/158/
        tag: matrix_hash
        """
        m, n, a, b = ac.read_list_ints()
        grid = []
        for _ in range(n):
            grid.extend([int(w) for w in ac.read_str()])
        mh = MatrixHash(m, n, grid)
        pre = set()
        for i in range(a - 1, m):
            for j in range(b - 1, n):
                pre.add(mh.query_sub(i, j, a, b))

        for _ in range(ac.read_int()):
            mat = []
            for _ in range(a):
                mat.extend([int(w) for w in ac.read_str()])
            cur = mh.query_matrix(a, b, mat)
            ac.st(1 if cur in pre else 0)
        return

    @staticmethod
    def ac_157(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/159/
        tag: tree_hash|tree_minimum_expression
        """

        def check(st):
            # 解码原始树的字符串表示，再tree_minimum_expression

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

            # 生成tree_minimum_expression
            n = ind + 1
            stack = [0]
            sub = [""] * n
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
                        lst.append("0" + sub[j] + "1")
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
    def lc_2851(s: str, t: str, k: int) -> int:
        """
        url: https://leetcode.cn/problems/string-transformation/
        tag: string_hash|kmp|matrix_dp|matrix_fast_power
        """
        mod = 10 ** 9 + 7
        n = len(s)
        sh1 = StringHash([ord(w) - ord("a") for w in t] + [26] + [ord(w) - ord("a") for w in s + s])
        sh2 = StringHash([ord(w) - ord("a") for w in t] + [26] + [ord(w) - ord("a") for w in s + s])
        target = (sh1.query(0, n - 1), sh2.query(0, n - 1))
        p = sum((sh1.query(i - n + 1, i), sh2.query(0, n - 1)) == target for i in range(2 * n, 3 * n))
        q = n - p
        mat = [[p - 1, p], [q, q - 1]]
        vec = [1, 0] if (sh1.query(n + 1, 2 * n), sh2.query(n + 1, 2 * n)) == target else [0, 1]
        res = MatrixFastPower().matrix_pow(mat, k, mod)
        ans = vec[0] * res[0][0] + vec[1] * res[0][1]
        return ans % mod

    @staticmethod
    def lg_p2852(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2852
        tag: binary_search|suffix_array|height|monotonic_queue|string_hash
        """

        def check(x):
            pre = defaultdict(int)
            for i in range(n):
                if i >= x - 1:
                    pre[(sh1.query(i - x + 1, i), sh2.query(i - x + 1, i))] += 1
            return max(pre.values()) >= k

        n, k = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]
        sh1 = StringHash(nums)
        sh2 = StringHash(nums)
        ans = BinarySearch().find_int_right(0, n, check)
        ac.st(ans)

        return

    @staticmethod
    def lg_p4656(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4656
        tag: string_hash|greedy
        """
        # string_hashgreedy选取

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
                # 从两边依次选取
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
                # 如果构成一对回文增| 2 否则增| 1
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
        """
        url: https://www.luogu.com.cn/problem/P6739
        tag: prefix_suffix|string_hash
        """
        n = ac.read_int()
        s = ac.read_str()
        if n % 2 == 0:
            ac.st("NOT POSSIBLE")
            return

        sh1 = StringHash([ord(w) - ord("a") for w in s])
        sh2 = StringHash([ord(w) - ord("a") for w in s])

        ans = dict()
        for i in range(n):
            if len(ans) > 1:
                break
            if i < n // 2:
                ss = (sh1.query(0, i - 1), sh2.query(0, i - 1))
                tt = (sh1.query(i + 1, n // 2), sh2.query(i + 1, n // 2))
                a = (ss[0] * sh1.pp[n // 2 - i] + tt[0]) % sh1.mod
                b = (ss[1] * sh2.pp[n // 2 - i] + tt[1]) % sh2.mod

                if sh1.query(n // 2 + 1, n - 1) == a and sh2.query(n // 2 + 1, n - 1) == b:
                    ans[(a, b)] = i

            elif i == n // 2:
                a, b = sh1.query(0, n // 2 - 1), sh2.query(0, n // 2 - 1)
                if sh1.query(n // 2 + 1, n - 1) == a and sh2.query(n // 2 + 1, n - 1) == b:
                    ans[(a, b)] = i
            else:
                ss = (sh1.query(n // 2, i - 1), sh2.query(n // 2, i - 1))
                tt = (sh1.query(i + 1, n - 1), sh2.query(i + 1, n - 1))
                a = (ss[0] * sh1.pp[n - 1 - i] + tt[0]) % sh1.mod
                b = (ss[1] * sh2.pp[n - 1 - i] + tt[1]) % sh2.mod

                if sh1.query(0, n // 2 - 1) == a and sh2.query(0, n // 2 - 1) == b:
                    ans[(a, b)] = i
        if not ans:
            ac.st("NOT POSSIBLE")
        elif len(ans) > 1:
            ac.st("NOT UNIQUE")
        else:
            i = list(ans.values())[0]
            if i >= n // 2:
                ac.st(s[:n // 2])
            elif i < n // 2:
                ac.st(s[-(n // 2):])
        return

    @staticmethod
    def lc_1948(paths: List[List[str]]) -> List[List[str]]:
        """
        url: https://leetcode.cn/problems/delete-duplicate-folders-in-system/
        tag: trie_like|tree_hash
        """
        # tree_hashtrie的子树编码

        dct = dict()  # 建树
        for path in paths:
            cur = dct
            for w in path:
                if w not in cur:
                    cur[w] = dict()
                cur = cur[w]
            cur["**"] = 1

        def dfs(node, cur_w):  # hash
            if not node:
                return tuple([0])

            state = tuple()
            for ww in sorted(node):
                if ww != "**" and ww != "##":
                    state += dfs(node[ww], ww)
            if state not in seen:
                seen[state] = len(seen) + 1
            node["##"] = seen[state]
            cnt[seen[state]] += 1
            return seen[state], cur_w

        seen = dict()
        cnt = Counter()
        dfs(dct, "")

        def dfs(node):
            if not node:
                return
            if cnt[node["##"]] > 1 and node["##"] > 1:
                return
            if pre:
                ans.append(pre[:])
            for ww in node:
                if ww != "##" and ww != "**":
                    pre.append(ww)
                    dfs(node[ww])
                    pre.pop()
            return

        # back_trace取出路径
        ans = []
        pre = []
        dfs(dct)
        return ans

    @staticmethod
    def lc_2261(nums: List[int], k: int, p: int) -> int:
        """
        url: https://leetcode.cn/problems/k-divisible-elements-subarrays/submissions/
        tag: string_hash
        """
        # string_hash对数组编码
        n = len(nums)
        pre = list(accumulate([int(num % p == 0) for num in nums], initial=0))
        p = [random.randint(26, 100), random.randint(26, 100)]
        mod = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]
        ans = set()
        for i in range(n):
            lst = [0, 0]
            for j in range(i, n):
                if pre[j + 1] - pre[i] <= k:
                    for x in range(2):
                        lst[x] = (lst[x] * p[x] + nums[j]) % mod[x]
                    ans.add((j - i + 1,) + tuple(lst))
                else:
                    break
        return len(ans)

    @staticmethod
    def lc_1316(text: str) -> int:
        """
        url: https://leetcode.cn/problems/distinct-echo-substrings
        tag: string_hash
        """
        n = len(text)
        sh1 = StringHash([ord(w) - ord("a") for w in text])
        sh2 = StringHash([ord(w) - ord("a") for w in text])

        ans = 0
        for x in range(1, n // 2 + 1):
            cur = set()
            for i in range(n - 2 * x + 1):
                ans1 = (sh1.query(i, i + x - 1), sh2.query(i, i + x - 1))
                ans2 = (sh1.query(i + x, i + 2 * x - 1), sh2.query(i + x, i + 2 * x - 1))
                if ans1 == ans2:
                    cur.add(ans1)
            ans += len(cur)
        return ans

    @staticmethod
    def lc_1923_1(n: int, paths: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/longest-common-subpath/
        tag: binary_search|string_hash
        """

        def check(x):
            pre = set()
            ind = 0
            for i in range(k):
                m = len(paths[i])
                cur = set()
                for j in range(ind, ind + m - x + 1):
                    cur.add((sh1.query(j, j + x - 1), sh2.query(j, j + x - 1)))
                if not i:
                    pre = cur
                else:
                    pre = pre.intersection(cur)
                ind += m
            return len(pre) > 0

        k = len(paths)
        lst = []
        for path in paths:
            lst.extend(path)
        n += 1
        sh1 = StringHash(lst)
        sh2 = StringHash(lst)
        ans = BinarySearch().find_int_right(0, min(len(p) for p in paths), check)
        return ans

    @staticmethod
    def lc_1923_2(n: int, paths: List[List[int]]) -> int:
        """
        url: https://leetcode.cn/problems/longest-common-subpath/
        tag: binary_search|string_hash
        """

        def check(x):
            pre = set()
            ind = 0
            for i in range(k):
                m = len(paths[i])
                cur = set()
                for j in range(ind, ind + m - x + 1):
                    cur.add(sh1.query(j, j + x - 1))
                if not i:
                    pre = cur
                else:
                    pre = pre.intersection(cur)
                if not pre:
                    return False
                ind += m
            return len(pre) > 0

        k = len(paths)
        lst = []
        for path in paths:
            lst.extend(path)
        sh1 = StringHashSingle(lst)
        ans = BinarySearch().find_int_right(0, min(len(p) for p in paths), check)
        return ans

    @staticmethod
    def lg_p3370_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3370
        tag: string_hash
        """
        ans = set()
        p1, p2 = [random.randint(26, 100), random.randint(26, 100)]
        mod1, mod2 = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]
        for _ in range(ac.read_int()):
            s = ac.read_str()
            x1 = x2 = 0
            for w in s:
                x1 = (x1 * p1 + ord(w)) % mod1
                x2 = (x2 * p2 + ord(w)) % mod2
            ans.add((x1, x2))
        ac.st(len(ans))
        return

    @staticmethod
    def lg_p3370_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3370
        tag: string_hash
        """
        ans = set()
        for _ in range(ac.read_int()):
            ans.add(hash(ac.read_str()))
        ac.st(len(ans))
        return

    @staticmethod
    def cf_514c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/514/problem/C
        tag: string_hash
        """

        n = 6 * 10 ** 5
        p = [random.randint(26, 100), random.randint(26, 100)]
        mod = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]
        pre = [[0] * (n + 1), [0] * (n + 1)]
        pp = [[1] * (n + 1), [1] * (n + 1)]
        for j in range(n):
            for i in range(2):
                pp[i][j + 1] = (pp[i][j] * p[i]) % mod[i]

        def query(x, y):
            if y < x:
                return 0, 0
            res = tuple((pre[i][y + 1] - pre[i][x] * pp[i][y - x + 1]) % mod[i] for i in range(2))
            return res

        ans = set()
        n, m = ac.read_list_ints()
        for _ in range(n):
            s = ac.read_str()
            k = len(s)
            lst = [ord(w) - ord("a") for w in s]
            for j, w in enumerate(lst):
                for i in range(2):
                    pre[i][j + 1] = (pre[i][j] * p[i] + w) % mod[i]

            for i in range(k):
                ll = query(0, i - 1)
                rr = query(i + 1, k - 1)
                for w in range(3):
                    if w != lst[i]:
                        cur = [0, 0]
                        for j in range(2):
                            cur[j] = ((ll[j] * p[j] + w) * pp[j][k - i - 1] + rr[j]) % mod[j]
                        ans.add((k, cur[0], cur[1]))

        for _ in range(m):
            s = ac.read_str()
            k = len(s)
            lst = [ord(w) - ord("a") for w in s]
            cur = [0, 0]
            for j, w in enumerate(lst):
                for i in range(2):
                    cur[i] = (cur[i] * p[i] + w) % mod[i]
            ac.st("YES" if (k, cur[0], cur[1]) in ans else "NO")
        return

    @staticmethod
    def cf_1200e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1200/problem/E
        tag: string_hash|kmp
        """
        ac.read_int()
        lst = ac.read_list_strs()
        n = sum(len(s) for s in lst)
        p = [random.randint(26, 100), random.randint(26, 100)]
        mod = [random.randint(10 ** 9 + 7, 2 ** 31 - 1), random.randint(10 ** 9 + 7, 2 ** 31 - 1)]
        pre = [[0] * (n + 1), [0] * (n + 1)]
        pp = [[1] * (n + 1), [1] * (n + 1)]
        for j in range(n):
            for i in range(2):
                pp[i][j + 1] = (pp[i][j] * p[i]) % mod[i]

        def query1(x, y):
            if y < x:
                return 0, 0
            res = tuple((pre[ii][y + 1] - pre[ii][x] * pp[ii][y - x + 1]) % mod[ii] for ii in range(2))
            return res

        def query2(x, y):
            if y < x:
                return 0, 0
            res = tuple((cur[ii][y + 1] - cur[ii][x] * pp[ii][y - x + 1]) % mod[ii] for ii in range(2))
            return res

        ans = []
        k = 0
        for word in lst:
            m = len(word)
            cur = [[0] * (m + 1), [0] * (m + 1)]
            inter = 0
            for j, w in enumerate(word):
                for i in range(2):
                    cur[i][j + 1] = (cur[i][j] * p[i] + ord(w)) % mod[i]
                if query1(k - j - 1, k - 1) == query2(0, j):
                    inter = j + 1
            for j in range(inter, m):
                w = word[j]
                ans.append(w)
                for i in range(2):
                    pre[i][k + 1] = (pre[i][k] * p[i] + ord(w)) % mod[i]
                k += 1
        ac.st("".join(ans))
        return

    @staticmethod
    def lc_1044(s: str) -> str:
        """
        url: https://leetcode.cn/problems/longest-duplicate-substring/
        tag: suffix_array|height|classical|string_hash
        """
        sh1 = StringHash([ord(w) - ord("a") for w in s])
        sh2 = StringHash([ord(w) - ord("a") for w in s])
        n = len(s)

        def check(x):
            return compute(x) > -1

        def compute(x):
            pre = set()
            for ii in range(n - x + 1):
                cur1, cur2 = sh1.query(ii, ii + x - 1), sh2.query(ii, ii + x - 1)
                if (cur1, cur2) in pre:
                    return ii
                pre.add((cur1, cur2))
            return -1

        length = BinarySearch().find_int_right(0, n, check)
        if length == 0:
            return ""
        i = compute(length)
        return s[i:i + length]

    @staticmethod
    def lc_1554(words: List[str]) -> bool:
        """
        url: https://leetcode.cn/problems/strings-differ-by-one-character/
        tag: string_hash|trie
        """
        pre = set()
        m = len(words[0])
        sh1 = StringHash([0] * m)
        sh2 = StringHash([0] * m)
        for word in words:
            lst = [ord(w) - ord("a") for w in word]
            for j, w in enumerate(lst):
                sh1.pre[j + 1] = (sh1.pre[j] * sh1.p + w) % sh1.mod
                sh2.pre[j + 1] = (sh2.pre[j] * sh2.p + w) % sh2.mod

            for j in range(m):
                ll = sh1.query(0, j - 1)
                rr = sh1.query(j + 1, m - 1)
                cur1 = ((ll * sh1.p + 26) * sh1.pp[m - j - 1] + rr) % sh1.mod

                ll = sh2.query(0, j - 1)
                rr = sh2.query(j + 1, m - 1)
                cur2 = ((ll * sh2.p + 26) * sh2.pp[m - j - 1] + rr) % sh2.mod

                if (cur1, cur2) in pre:
                    return True
                pre.add((cur1, cur2))
        return False

    @staticmethod
    def lc_2156(s: str, p: int, modulo: int, k: int, hash_value: int) -> str:
        """
        url: https://leetcode.cn/problems/find-substring-with-given-hash-value
        tag: string_hash|reverse_order
        """
        ans = -1
        n = len(s)
        pp = pow(p, k - 1, modulo)
        post = 0
        for i in range(n - 1, -1, -1):
            post = post * p + ord(s[i]) - ord("a") + 1
            post %= modulo
            if post == hash_value and i + k - 1 <= n - 1:
                ans = i
            if i + k - 1 <= n - 1:
                post -= (ord(s[i + k - 1]) - ord("a") + 1) * pp
                post %= modulo
        return s[ans: ans + k]

    @staticmethod
    def library_check_1(ac=FastIO()):
        """
        url: https://ac.nowcoder.com/acm/contest/64384/D
        tag: string_hash|implemention
        """
        n, m, k = ac.read_list_ints()
        s = ac.read_str()
        sh1 = StringHash([int(w) for w in s])
        sh2 = StringHash([int(w) for w in s])
        ind = defaultdict(list)
        for i in range(m - 1, n):
            cur = (sh1.query(i - m + 1, i), sh2.query(i - m + 1, i))
            ind[cur].append(i)
        ans = 0
        for lst in ind.values():
            cur = 0
            pre = -m
            for i in lst:
                if i - pre >= m:
                    cur += 1
                    pre = i
            ans += cur == k
        ac.st(ans)
        return

    @staticmethod
    def cf_452f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/452/problem/F
        tag: segment_tree_hash|string_hash|point_set|range_hash|range_reverse
        """
        n = ac.read_int()
        tree1 = PointSetRangeHashReverse(n)  # TLE
        tree2 = PointSetRangeHashReverse(n)
        nums = ac.read_list_ints_minus_one()
        for num in nums:
            tree1.point_set(num, num, 1)
            tree2.point_set(num, num, 1)
            if num == 0 or num == n - 1:
                continue
            length = min(num + 1, n - num)
            cur1 = [tree1.range_hash(num - length + 1, num), tree2.range_hash(num - length + 1, num)]
            cur2 = [tree1.range_hash_reverse(num, num + length - 1), tree2.range_hash_reverse(num, num + length - 1)]
            if cur1 != cur2:
                ac.yes()
                break
        else:
            ac.no()
        return

    @staticmethod
    def cf_580e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/580/problem/E
        tag: segment_tree_hash|range_change|range_hash_reverse|circular_section
        """
        n, m, k = ac.read_list_ints()
        tree1 = RangeSetRangeHashReverse(n, 10)
        tree2 = RangeSetRangeHashReverse(n, 10)
        s = ac.read_str()
        tree1.build([int(w) for w in s])
        tree2.build([int(w) for w in s])
        for _ in range(m + k):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                l, r, c = lst[1:]
                tree1.range_set(l - 1, r - 1, c)
                tree2.range_set(l - 1, r - 1, c)
            else:
                l, r, d = lst[1:]
                if d == r - l + 1:
                    ac.yes()
                    continue
                else:
                    if tree1.range_hash(l - 1, r - d - 1) == tree1.range_hash(l + d - 1, r - 1):
                        if tree2.range_hash(l - 1, r - d - 1) == tree2.range_hash(l + d - 1, r - 1):
                            ac.yes()
                        else:
                            ac.no()
                    else:
                        ac.no()
        return

    @staticmethod
    def lc_2977(source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:

        """
        url: https://leetcode.cn/problems/minimum-cost-to-convert-string-ii
        tag: string_hash|dp|dijkstra|trie
        """

        sh1 = StringHash([ord(w) - ord("a") for w in source + target])
        sh2 = StringHash([ord(w) - ord("a") for w in source + target])

        has = dict()

        nodes = set(original + changed)
        for string in nodes:
            lst = [len(string)]

            p, mod = sh1.p, sh1.mod
            state = 0
            for w in string:
                state *= p
                state += ord(w) - ord("a")
                state %= mod
            lst.append(state)

            p, mod = sh2.p, sh2.mod
            state = 0
            for w in string:
                state *= p
                state += ord(w) - ord("a")
                state %= mod
            lst.append(state)

            has[tuple(lst)] = string

        ind = {x: i for i, x in enumerate(nodes)}
        m = len(ind)
        dct = [[] for _ in range(m)]
        for x, y, z in zip(original, changed, cost):
            x = ind[x]
            y = ind[y]
            dct[x].append([y, z])
        dis = []
        for x in range(m):
            dis.append(Dijkstra().get_shortest_path(dct, x))

        n = len(source)
        dp = [math.inf] * (n + 1)

        exist = defaultdict(list)
        for s in original:
            exist[s[-1]].append(s)

        dp[0] = 0
        for i in range(n):
            if source[i] == target[i]:
                dp[i + 1] = dp[i]
            for w in exist[source[i]]:
                j = i - len(w) + 1
                s1 = (i - j + 1, sh1.query(j, i), sh2.query(j, i))
                t1 = (i - j + 1, sh1.query(j + n, i + n), sh2.query(j + n, i + n))
                if s1 in has and t1 in has:
                    x, y = ind[has[s1]], ind[has[t1]]
                    z = dis[x][y]
                    if dp[j] + z < dp[i + 1]:
                        dp[i + 1] = dp[j] + z

        return dp[-1] if dp[-1] < math.inf else -1

    @staticmethod
    def ac_139(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/139/
        tag: matrix_hash
        """
        p1 = random.randint(26, 100)
        p2 = random.randint(26, 100)
        mod1 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)
        mod2 = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

        def compute(ls):
            res1 = 0
            for num in ls:
                res1 *= p1
                res1 += num
                res1 %= mod1
            res2 = 0
            for num in ls:
                res2 *= p2
                res2 += num
                res2 %= mod2
            return res1, res2

        def check():
            res = []
            for ii in range(6):
                cu = tuple(lst[ii:] + lst[:ii])
                res.append(compute(cu))
                cu = tuple(lst[:ii + 1][::-1] + lst[ii + 1:][::-1])
                res.append(compute(cu))
            return res

        n = ac.read_int()
        pre = set()
        for _ in range(n):
            lst = ac.read_list_ints()
            now = check()
            if any(cur in pre for cur in now):
                ac.st("Twin snowflakes found.")
                break
            for cur in now:
                pre.add(cur)
        else:
            ac.st("No two snowflakes are alike.")
        return

    @staticmethod
    def library_check_2(ac=FastIO()):

        """
        url: https://www.luogu.com.cn/problem/UVA11019
        tag: matrix_hash|string_hash
        """
        for _ in range(ac.read_int()):  # TLE
            m, n = ac.read_list_ints()
            grid = []
            for _ in range(n):
                grid.extend([ord(w) - ord("a") for w in ac.read_str()])

            mh = MatrixHash(m, n, grid)
            a, b = ac.read_list_ints()
            mat = []
            for _ in range(a):
                mat.extend([ord(w) - ord("a") for w in ac.read_str()])
            cur = mh.query_matrix(a, b, mat)
            ans = 0
            for i in range(a - 1, n):
                for j in range(b - 1, m):
                    if mh.query_sub(i, j, a, b) == cur:
                        ans += 1
            ac.st(ans)
        return

    @staticmethod
    def library_check_3(ac=FastIO()):
        """
        url: https://ac.nowcoder.com/acm/problem/51003
        tag: matrix_hash|string_hash
        """

        m, n, a, b = ac.read_list_ints()
        grid = []
        for _ in range(n):
            grid.extend([int(w) for w in ac.read_str()])
        mh = MatrixHash(m, n, grid)
        pre = set()
        for i in range(a - 1, m):
            for j in range(b - 1, n):
                pre.add(mh.query_sub(i, j, a, b))

        for _ in range(ac.read_int()):
            mat = []
            for _ in range(a):
                mat.extend([int(w) for w in ac.read_str()])
            cur = mh.query_matrix(a, b, mat)
            ac.st(1 if cur in pre else 0)
        return

    @staticmethod
    def lg_p2601(ac=FastIO()):

        """
        url: https://www.luogu.com.cn/problem/P2601
        tag: matrix_hash|string_hash
        """
        m, n = ac.read_list_ints()
        lst = []
        for _ in range(m):
            lst.extend(ac.read_list_ints())
        mh = MatrixHashReverse(m, n, lst)

        def check1(x):

            res1 = mh.query_left_up(i + x - 1, j + x - 1, 2 * x - 1, 2 * x - 1)
            res2 = mh.query_right_up(i + x - 1, j - x + 1, 2 * x - 1, 2 * x - 1)
            res3 = mh.query_left_down(i - x + 1, j + x - 1, 2 * x - 1, 2 * x - 1)
            res4 = mh.query_right_down(i - x + 1, j - x + 1, 2 * x - 1, 2 * x - 1)
            return res1 == res2 == res3 == res4

        def check2(x):

            res1 = mh.query_left_up(i + x, j + x, 2 * x, 2 * x)
            res2 = mh.query_right_up(i + x, j - x + 1, 2 * x, 2 * x)
            res3 = mh.query_left_down(i - x + 1, j + x, 2 * x, 2 * x)
            res4 = mh.query_right_down(i - x + 1, j - x + 1, 2 * x, 2 * x)
            return res1 == res2 == res3 == res4

        bs = BinarySearch()
        ans = i = j = 0

        for i in range(m):
            for j in range(n):
                y = min(i + 1, m - i, j + 1, n - j)
                ans += bs.find_int_right(0, y, check1)
                y = min(i + 1, j + 1, m - i - 1, n - j - 1)
                ans += bs.find_int_right(0, y, check2)
        ac.st(ans)
        return

    @staticmethod
    def lg_p4824(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4824
        tag: string_hash|stack|implemention
        """

        s = ac.read_str()
        t = ac.read_str()
        m, n = len(s), len(t)
        sh = StringHash([ord(w) - ord("a") for w in s + t])
        del t
        target = sh.query(m, m + n - 1)
        i = 0
        stack = []
        for w in s:
            x = ord(w) - ord("a")
            stack.append(w)
            sh.pre[i + 1] = (sh.pre[i] * sh.p + x) % sh.mod
            i += 1
            if i >= n and sh.query(i - n, i - 1) == target:
                i -= n
                for _ in range(n):
                    stack.pop()
        ac.st("".join(stack))
        return

    @staticmethod
    def lg_p4503(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4503
        tag: string_hash
        """

        ind = dict()
        words = []
        for i in range(10):
            ind[str(i)] = i
            words.append(str(i))
        for i in range(26):
            ind[chr(i + ord("a"))] = 10 + i
            words.append(chr(i + ord("a")))
        for i in range(26):
            ind[chr(i + ord("A"))] = 36 + i
            words.append(chr(i + ord("A")))
        ind["_"] = 62
        ind["@"] = 63
        words.extend(["_", "@"])

        n, ll, s = ac.read_list_ints()
        sh1 = StringHash([0] * ll)
        sh2 = StringHash([0] * ll)
        cnt = dict()
        ans = 0
        for _ in range(n):
            st = [ind[w] + 1 for w in ac.read_str()]
            for j in range(ll):
                sh1.pre[j + 1] = (sh1.pre[j] * sh1.p + st[j]) % sh1.mod
                sh2.pre[j + 1] = (sh2.pre[j] * sh2.p + st[j]) % sh2.mod
            for j in range(ll):
                pre1 = sh1.pre[j]
                post1 = sh1.query(j + 1, ll - 1)
                pre2 = sh2.pre[j]
                post2 = sh2.query(j + 1, ll - 1)
                right = ll - j - 1
                cur1 = (pre1 * sh1.pp[right + 1] + post1) % sh1.mod
                cur2 = (pre2 * sh2.pp[right + 1] + post2) % sh2.mod
                ans += cnt.get((cur1, cur2), 0)
                cnt[(cur1, cur2)] = cnt.get((cur1, cur2), 0) + 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p3538(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3538
        tag: string_hash|prime_factor|brute_force|circular_section
        """
        n = ac.read_int()
        lst = [ord(w) - ord("a") for w in ac.read_str()]
        sh1 = StringHash(lst)
        sh2 = StringHash(lst)  # TLE

        def check(cur):
            pre1 = sh1.query(a, b - cur)
            post1 = sh1.query(a + cur, b)

            pre2 = sh2.query(a, b - cur)
            post2 = sh2.query(a + cur, b)
            return pre1 == post1 and pre2 == post2

        pf = PrimeFactor(n)
        for _ in range(ac.read_int()):
            a, b = ac.read_list_ints_minus_one()
            length = b - a + 1
            ans = length
            while length > 1:
                p = pf.min_prime[length]
                if check(ans // p):
                    ans //= p
                length //= p
            ac.st(ans)
        return

    @staticmethod
    def cf_7d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/7/D
        tag: string_hash|palindrome|classical
        """
        p = 131
        mod = 10 ** 9 + 7
        s = ac.read_str()
        n = len(s)
        pre = rev = 0
        pp = 1
        dp = [0] * (n + 1)
        for i in range(n):
            x = ord(s[i]) - ord("a")
            pre = (pre * p + x) % mod
            rev = (x * pp + rev) % mod
            pp = (pp * p) % mod
            if pre != rev:
                continue
            dp[i + 1] = dp[(i + 1) // 2] + 1
        ac.st(sum(dp))
        return

    @staticmethod
    def cf_835d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/835/D
        tag: palindrome|string_hash
        """
        s = ac.read_str()
        lst = [ord(w) - ord("a") for w in s]
        n = len(s)
        dp = [0] * (n + 1)
        ans = [0] * (n + 1)
        p = 131
        mod = 10 ** 9 + 7
        for i in range(n):
            dp[i] = 1
            pp = p
            pre = rev = lst[i]
            ans[1] += 1
            for j in range(i + 1, n):
                pre = (pre * p + lst[j]) % mod
                rev = (lst[j] * pp + rev) % mod
                pp = (pp * p) % mod
                if pre == rev:
                    dp[j] = dp[i + (j - i + 1) // 2 - 1] + 1
                    ans[dp[j]] += 1
                else:
                    dp[j] = 0
        for i in range(n - 1, -1, -1):
            ans[i] += ans[i + 1]
        ac.lst(ans[1:])
        return

    @staticmethod
    def lc_100208(words: List[str]) -> int:
        """
        url: https://leetcode.com/contest/weekly-contest-385/problems/count-prefix-and-suffix-pairs-ii/
        tag: string_hash|brute_force
        """
        ans = 0
        st = "".join(words)
        sh1 = StringHash([ord(w) - ord("a") for w in st])
        sh2 = StringHash([ord(w) - ord("a") for w in st])
        pre = defaultdict(int)
        length = 0
        for word in words:
            m = len(word)
            for i in range(1, m + 1):
                prefix = (sh1.query(length, length + i - 1), sh2.query(length, length + i - 1))
                suffix = (sh1.query(length + m - 1 - i + 1, length + m - 1),
                          sh2.query(length + m - 1 - i + 1, length + m - 1))
                if prefix == suffix:
                    ans += pre[prefix]

            prefix = (sh1.query(length, length + m - 1), sh2.query(length, length + m - 1))
            pre[prefix] += 1
            length += m
        return ans

    @staticmethod
    def abc_331f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc331/tasks/abc331_f
        tag: point_set|range_hash_reverse|palindrome|classical
        """
        n, q = ac.read_list_ints()
        tree1 = PointSetRangeHashReverse(n)
        tree2 = PointSetRangeHashReverse(n)
        s = ac.read_str()
        tree1.build([ord(w) - ord("a") for w in s])
        tree2.build([ord(w) - ord("a") for w in s])
        for _ in range(q):
            lst = ac.read_list_strs()
            if lst[0] == "1":
                x, c = lst[1:]
                x = int(x)
                c = ord(c) - ord("a")
                tree1.point_set(x - 1, x - 1, c)
                tree2.point_set(x - 1, x - 1, c)
            else:
                ll, rr = [int(w) - 1 for w in lst[1:]]
                cur1 = (tree1.range_hash(ll, rr), tree2.range_hash(ll, rr))
                cur2 = (tree1.range_hash_reverse(ll, rr), tree2.range_hash_reverse(ll, rr))
                if cur1 == cur2:
                    ac.yes()
                else:
                    ac.no()
        return

    @staticmethod
    def abc_310c(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc310/tasks/abc310_c
        tag: string_hash|classical
        """
        n = ac.read_int()
        pre = 0
        sh = StringHashSingle([150])
        ans = set()
        for i in range(n):
            lst = [ord(w) - ord("a") for w in ac.read_str()]
            cur = len(lst)
            bb, aa = sh.check(lst)
            dd, cc = sh.check(lst[::-1])
            tp = [aa] + sorted([bb, dd])
            ans.add(tuple(tp))
            pre += cur
        ac.st(len(ans))
        return

    @staticmethod
    def abc_353e(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc353/tasks/abc353_e
        tag: string_hash|trie
        """
        ac.read_int()
        words = ac.read_list_strs()
        ans = 0
        p = random.randint(150, 150 * 2)
        mod = random.getrandbits(64)
        cnt = defaultdict(int)
        for word in words:
            hash_x = 0
            for i, w in enumerate(word):
                hash_x = (hash_x * p + ord(w) - ord("a")) % mod
                cur = cnt[(i, hash_x)]
                ans -= i * cur
                ans += (i + 1) * cur
                cnt[(i, hash_x)] += 1
        ac.st(ans)
        return

    @staticmethod
    def cf_1977d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1977/problem/D
        tag: string_hash|brute_force|brain_teaser|classical
        """
        for _ in range(ac.read_int()):
            m, n = ac.read_list_ints()
            grid = [ac.read_str() for _ in range(m)]
            dct = defaultdict(set)
            sh = StringHashSingleBuild(m)
            for j in range(n):
                lst = [int(grid[i][j]) for i in range(m)]
                sh.build(lst)
                for i in range(m):
                    left = sh.query(0, i - 1)
                    right = sh.query(i + 1, m - 1)
                    mid = 1 - int(grid[i][j])
                    cur = ((left * sh.p + mid) * sh.pp[m - i - 1] + right) % sh.mod
                    dct[cur].add(j)
            res = -1
            for k in dct:
                if res == -1 or len(dct[k]) > len(dct[res]):
                    res = k
            ac.st(len(dct[res]))
            j = list(dct[res])[0]
            lst = [int(grid[i][j]) for i in range(m)]
            sh.build(lst)
            for i in range(m):
                left = sh.query(0, i - 1)
                right = sh.query(i + 1, m - 1)
                mid = 1 - int(grid[i][j])
                cur = ((left * sh.p + mid) * sh.pp[m - i - 1] + right) % sh.mod
                if cur == res:
                    ans = [grid[i][j] for i in range(m)]
                    ans[i] = "1" if grid[i][j] == "0" else "0"
                    ac.st("".join(ans))
                    break
        return

    @staticmethod
    def cf_1418g(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1418/G
        tag: string_hash|random_hash|classical|two_pointers
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        rd = [random.getrandbits(64) for _ in range(5 * 10 ** 5 + 1)]
        ac.get_random_seed()
        pre_hash = [0] * (n + 1)
        cnt = defaultdict(int)
        for i in range(n):
            pre_hash[i + 1] = pre_hash[i]
            num = nums[i]
            pre_hash[i + 1] -= cnt[num] * rd[num]
            cnt[num] = (cnt[num] + 1) % 3
            pre_hash[i + 1] += cnt[num] * rd[num]

        pre = defaultdict(int)
        pre[pre_hash[0]] = 1
        cur = defaultdict(int)
        ans = i = 0
        for j in range(n):
            cur[nums[j]] += 1
            while cur[nums[j]] > 3:
                pre[pre_hash[i]] -= 1
                cur[nums[i]] -= 1
                i += 1
            ans += pre[pre_hash[j + 1]]
            pre[pre_hash[j + 1]] += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p4503(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4503
        tag: string_hash|random_hash
        """
        n, ll, ss = ac.read_list_ints()  # MLE
        seed = [random.getrandbits(64) for _ in range(n)]
        nums = [[0] * (ll + 1) for _ in range(n)]
        for i in range(n):
            x = 0
            s = ac.read_str()
            for j in range(ll):
                nums[i][j] = ord(s[j])
                x += ord(s[j]) * seed[j]
            nums[i][ll] = x
        ans = 0
        for j in range(ll):
            pre = dict()
            for i in range(n):
                val = nums[i][-1] - nums[i][j] * seed[j]
                ans += pre.get(val, 0)
                pre[val] = pre.get(val, 0) + 1
        ac.st(ans)
        return

    @staticmethod
    def abc_367f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc367/tasks/abc367_f
        tag: random_seed|random_hash
        """
        n, q = ac.read_list_ints()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        seed = [random.getrandbits(64) for _ in range(2 * 10 ** 5 + 1)]
        pre_a = ac.accumulate([seed[x] for x in a])
        pre_b = ac.accumulate([seed[x] for x in b])
        for _ in range(q):
            l1, r1, l2, r2 = ac.read_list_ints_minus_one()
            if pre_a[r1 + 1] - pre_a[l1] == pre_b[r2 + 1] - pre_b[l2] and r1 - l1 == r2 - l2:
                ac.yes()
            else:
                ac.no()
        return

    @staticmethod
    def lc_3327(parent: List[int], s: str) -> List[bool]:
        """
        url: https://leetcode.cn/problems/check-if-dfs-strings-are-palindromes/
        tag: dfs_order|manacher|palindrome|string_hash_single|classical
        """
        n = len(parent)
        graph = UnWeightedTree(n)
        for i in range(n - 1, 0, -1):
            graph.add_directed_edge(parent[i], i)
        graph.dfs_order(0)
        lst = [ord(s[i]) - ord("a") for i in graph.order_to_node]
        sh = StringHashSingle(lst + lst[::-1])
        ans = []
        for i in range(n):
            ans.append(sh.query(graph.start[i], graph.end[i]) == sh.query(n + n - 1 - graph.end[i], n + n - 1 - graph.start[i]))
        return ans
