"""

Algorithm：construction
Description：greedy|sort|construction|specific_plan

====================================LeetCode====================================
280（https://leetcode.cn/problems/wiggle-sort/）construction|sort|odd_even
2663（https://leetcode.cn/problems/lexicographically-smallest-beautiful-string/）greedy|construction|palindrome_substring|lexicographical_order
1982（https://leetcode.cn/problems/find-array-given-subset-sums/）construction
1253（https://leetcode.cn/problems/reconstruct-a-2-row-binary-matrix/）construction|greedy|brain_teaser
2573（https://leetcode.cn/problems/find-the-string-with-lcp/）lcp|construction|union_find

=====================================LuoGu======================================
P8846（https://www.luogu.com.cn/problem/P8846）greedy|construction
P2902（https://www.luogu.com.cn/problem/P2902）construction
P5823（https://www.luogu.com.cn/problem/P5823）construction
P7383（https://www.luogu.com.cn/problem/P7383）greedy|construction
P7947（https://www.luogu.com.cn/problem/P7947）greedy|construction|product_n_sum_k|prime_factorization
P9101（https://www.luogu.com.cn/problem/P9101）construction|directed_graph|no_circe
P8976（https://www.luogu.com.cn/problem/P8976）brute_force|construction
P8910（https://www.luogu.com.cn/problem/P8910）permutation_circle|construction
P8880（https://www.luogu.com.cn/problem/P8880）brain_teaser|construction|odd_even
P2902（https://www.luogu.com.cn/problem/P2902）construction
P8248（https://www.luogu.com.cn/problem/P8248）dfs|back_trace|brute_force
P7814（https://www.luogu.com.cn/problem/P7814）construction
P7567（https://www.luogu.com.cn/problem/P7567）construction|classical|guess_table
P8683（https://www.luogu.com.cn/problem/P8683）construction

===================================CodeForces===================================
1396A（https://codeforces.com/problemset/problem/1396/A）greedy|construction
1133F2（https://codeforces.com/contest/1133/problem/F2）mst|construction|union_find
1118C（https://codeforces.com/contest/1118/problem/C）construction|matrix_rotate|implemention
1118E（https://codeforces.com/problemset/problem/1118/E）implemention|greedy|construction
960C（https://codeforces.com/problemset/problem/960/C）greedy|construction
1793B（https://codeforces.com/contest/1793/problem/B）brain_teaser|greedy|construction
1375D（https://codeforces.com/problemset/problem/1375/D）mex|construction|sorting
1348D（https://codeforces.com/problemset/problem/1348/D）bin|construction
1554D（https://codeforces.com/problemset/problem/1554/D）construction|floor
1788C（https://codeforces.com/problemset/problem/1788/C）construction
1367D（https://codeforces.com/problemset/problem/1367/D）reverse_thinking|implemention|construction|classical
1485D（https://codeforces.com/problemset/problem/1485/D）data_range|construction
1722G（https://codeforces.com/problemset/problem/1722/G）odd_even|xor_property|construction
1822D（https://codeforces.com/contest/1822/problem/D）construction|prefix_sum|mod|permutation
1509D（https://codeforces.com/contest/1509/problem/D）lcs|shortest_common_hypersequence|construction|data_range|O(n)|pigeonhole_principle
1473C（https://codeforces.com/contest/1473/problem/C）brain_teaser|s1s2..sn..s2s1
1469D（https://codeforces.com/contest/1469/problem/D）square|ceil|greedy|implemention
1478B（https://codeforces.com/contest/1478/problem/B）brute_force|bag_dp|construction
1682B（https://codeforces.com/contest/1682/problem/B）bitwise_and|construction|permutation_circle
1823D（https://codeforces.com/contest/1823/problem/D）greedy|construction|palindrome
1352G（https://codeforces.com/contest/1352/problem/G）construction|odd_even
1352F（https://codeforces.com/contest/1352/problem/G）construction
1003E（https://codeforces.com/contest/1003/problem/E）construction|tree_diameter|classical
1005F（https://codeforces.com/contest/1005/problem/F）construction|shortest_path_spanning_tree|classical|dfs|specific_plan
1092E（https://codeforces.com/contest/1092/problem/E）construction|tree_diameter|classical|greedy
1141G（https://codeforces.com/problemset/problem/1141/G）construction|dfs|color_method|greedy|classical
1144F（https://codeforces.com/contest/1144/problem/F）construction|color_method|classical|bipartite
1157E（https://codeforces.com/contest/1157/problem/E）construction|greedy|sorted_list
1157D（https://codeforces.com/contest/1157/problem/D）construction|greedy
1196E（https://codeforces.com/contest/1196/problem/E）construction|greedy
1213E（https://codeforces.com/contest/1213/problem/E）construction|brute_force
1294F（https://codeforces.com/contest/1294/problem/F）classical|tree_diameter|construction
1311E（https://codeforces.com/contest/1311/problem/E）construction|2-tree
1343F（https://codeforces.com/contest/1343/problem/F）construction|data_range|brain_teaser
1360G（https://codeforces.com/contest/1360/problem/G）construction
1385E（https://codeforces.com/contest/1385/problem/E）construction|topological_sort|classical|undirected|directed
1475F（https://codeforces.com/contest/1475/problem/F）construction|matrix_reverse
1551D2（https://codeforces.com/contest/1551/problem/D2）construction|domino
1714F（https://codeforces.com/contest/1714/problem/F）construction|tree
1702F（https://codeforces.com/contest/1702/problem/F）construction|brain_teaser
1772F（https://codeforces.com/contest/1772/problem/F）construction
1899F（https://codeforces.com/contest/1899/problem/F）construction
1923C（https://codeforces.com/contest/1923/problem/C）construction
1968E（https://codeforces.com/contest/1968/problem/E）construction
1973C（https://codeforces.com/contest/1973/problem/C）construction
1974D（https://codeforces.com/contest/1974/problem/D）construction
1978C（https://codeforces.com/contest/1978/problem/C）construction
1338C（https://codeforces.com/contest/1338/problem/B）construction|tree_xor
1450C2（https://codeforces.com/contest/1450/problem/C2）construction
1854A1（https://codeforces.com/problemset/problem/1854/A1）construction
1854A2（https://codeforces.com/problemset/problem/1854/A2）construction
1416B（https://codeforces.com/problemset/problem/1416/B）construction
1217D（https://codeforces.com/problemset/problem/1217/D）construction|observation|classical
1758D（https://codeforces.com/problemset/problem/1758/D）construction
1268B（https://codeforces.com/problemset/problem/1268/B）construction
1552D（https://codeforces.com/problemset/problem/1552/D）construction
1364D（https://codeforces.com/problemset/problem/1364/D）dfs_tree|construction|independent_set|union_find|undirected_circle|undirected_local_shortest_circle
949A（https://codeforces.com/problemset/problem/949/A）observation|construction
1809C（https://codeforces.com/problemset/problem/1809/C）construction|diff_array|reverse_pair
1481D（https://codeforces.com/problemset/problem/1481/D）observation|construction
1658C（https://codeforces.com/problemset/problem/1658/C）construction
1861D（https://codeforces.com/problemset/problem/1861/D）observation|construction
1951D（https://codeforces.com/problemset/problem/1951/D）construction
1187C（https://codeforces.com/problemset/problem/1187/C）build_graph|brain_teaser|construction
1304D（https://codeforces.com/problemset/problem/1304/D）construction
2001C（https://codeforces.com/contest/2001/problem/C）construction|union_find|observation
1630A（https://codeforces.com/problemset/problem/1630/A）construction|bit_operation
1965B（https://codeforces.com/problemset/problem/1965/B）construction
1991D（https://codeforces.com/problemset/problem/1991/D）construction
1264B（https://codeforces.com/problemset/problem/1264/B）construction
1603B（https://codeforces.com/problemset/problem/1603/B）construction
1848C（https://codeforces.com/problemset/problem/1848/C）construction
735D（https://codeforces.com/problemset/problem/735/D）construction|number_theory
1616D（https://codeforces.com/problemset/problem/1616/D）construction|observation|brain_teaser|average_trick
1545A（https://codeforces.com/problemset/problem/1545/A）construction|observation
1635D（https://codeforces.com/contest/1635/problem/D）fibonacci|brain_teaser|construction
552C（https://codeforces.com/problemset/problem/552/C）construction|math|divide_and_conquer
1371D（https://codeforces.com/problemset/problem/1371/D）construction
1332D（https://codeforces.com/problemset/problem/1332/D）construction|brain_teaser
576A（https://codeforces.com/problemset/problem/576/A）eratosthenes_sieve|construction
348A（https://codeforces.com/problemset/problem/348/A）construction
1798C（https://codeforces.com/problemset/problem/1798/C）construction
1380D（https://codeforces.com/problemset/problem/1380/D）construction
1553D（https://codeforces.com/problemset/problem/1553/D）construction
1537E2（https://codeforces.com/problemset/problem/1537/E2）construction
675C（https://codeforces.com/problemset/problem/675/C）construction
1208C（https://codeforces.com/problemset/problem/1208/C）construction
1924A（https://codeforces.com/problemset/problem/1924/A）construction

====================================AtCoder=====================================
AGC007B（https://atcoder.jp/contests/agc007/tasks/agc007_b）brain_teaser|math|construction
ARC086B（https://atcoder.jp/contests/abc081/tasks/arc086_b）greedy|construction|classification_discussion
ARC093B（https://atcoder.jp/contests/abc092/tasks/arc093_b）brain_teaser|construction
ABC126F（https://atcoder.jp/contests/abc126/tasks/abc126_f）brain_teaser|construction|xor_property
ABC109D（https://atcoder.jp/contests/abc109/tasks/abc109_d）odd_even|construction
ABC345F（https://atcoder.jp/contests/abc345/tasks/abc345_f）construction|union_find|greedy|implemention
ABC299E（https://atcoder.jp/contests/abc299/tasks/abc299_e）construction|bfs
ABC251F（https://atcoder.jp/contests/abc251/tasks/abc251_f）construction|dfs|bfs|classical
ABC251D（https://atcoder.jp/contests/abc251/tasks/abc251_d）construction|brute_force|brain_teaser
ABC239F（https://atcoder.jp/contests/abc239/tasks/abc239_f）implemention|construction|greedy|brain_teaser|union_find
ABC233F（https://atcoder.jp/contests/abc233/tasks/abc233_f）graph|union_find|construction|mst|brain_teaser|classical
ABC231D（https://atcoder.jp/contests/abc231/tasks/abc231_d）union_find|construction
ABC225C（https://atcoder.jp/contests/abc225/tasks/abc225_c）construction
ABC362F（https://atcoder.jp/contests/abc362/tasks/abc362_f）construction|greedy|observation

====================================AtCoder=====================================
1（https://www.codechef.com/problems/ENVPILE）bfs|construction|classical


"""
import math
from collections import deque, Counter, defaultdict
from heapq import heappush, heappop, heapify
from typing import List

from src.graph.union_find.template import UnionFind
from src.mathmatics.number_theory.template import NumFactor
from src.utils.fast_io import FastIO, inf


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1478b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1478/problem/B
        tag: brute_force|bag_dp|construction|brain_teaser|classical
        """
        for _ in range(ac.read_int()):
            q, d = ac.read_list_ints()
            queries = ac.read_list_ints()
            ceil = 10 * d + 9
            dp = [0] * (ceil + 1)
            dp[0] = 1
            for i in range(1, ceil + 1):
                if str(d) in str(i):
                    for j in range(i, ceil + 1):
                        if dp[j - i]:
                            dp[j] = 1
            for num in queries:
                if num >= 10 * d + 9 or dp[num]:
                    ac.yes()
                else:
                    ac.no()
        return

    @staticmethod
    def cf_1367d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1367/D
        tag: reverse_thinking|implemention|construction
        """
        for _ in range(ac.read_int()):
            s = ac.read_str()
            m = ac.read_int()
            nums = ac.read_list_ints()
            ans = [""] * m
            lst = deque(sorted(list(s), reverse=True))
            while max(nums) >= 0:
                zero = [i for i in range(m) if nums[i] == 0]
                k = len(zero)
                while len(set(list(lst)[:k])) != 1:
                    lst.popleft()
                for i in zero:
                    nums[i] = -1
                    ans[i] = lst.popleft()
                while lst and lst[0] == ans[zero[0]]:
                    lst.popleft()
                for i in range(m):
                    if nums[i] != -1:
                        nums[i] -= sum(abs(i - j) for j in zero)
            ac.st("".join(ans))
        return

    @staticmethod
    def cf_1788c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1788/C
        tag: construction
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            if n % 2:
                ac.yes()
                x = n // 2
                for i in range(1, n + 1):
                    if i <= x:
                        ac.lst([i, i + n + x + 1])
                    else:
                        ac.lst([i, i - x + n])
            else:
                ac.no()
        return

    @staticmethod
    def lc_280(nums: List[int]) -> None:
        """
        url: https://leetcode.cn/problems/wiggle-sort/
        tag: construction|sort|odd_even|classical
        """
        nums.sort()
        n = len(nums)
        ans = [0] * n
        j = n - 1
        for i in range(1, n, 2):
            ans[i] = nums[j]
            j -= 1
        j = 0
        for i in range(0, n, 2):
            ans[i] = nums[j]
            j += 1
        for i in range(n):
            nums[i] = ans[i]
        return

    @staticmethod
    def lc_1982(n: int, sums: List[int]) -> List[int]:
        """
        url: https://leetcode.cn/problems/find-array-given-subset-sums/
        tag: construction|brain_teaser|classical
        """
        low = min(sums)
        if low < 0:
            sums = [num - low for num in sums]

        cnt = Counter(sums)
        lst = sorted(cnt.keys())
        cnt[0] -= 1
        ans = []
        pre = defaultdict(int)
        pre_sum = []
        for _ in range(n):
            for num in lst:
                if cnt[num] > pre[num]:
                    ans.append(num)
                    for p in pre_sum[:]:
                        pre[p + num] += 1
                        pre_sum.append(p + num)
                    pre[num] += 1
                    pre_sum.append(num)
                    break

        for i in range(1 << n):
            cur = [j for j in range(n) if i & (1 << j)]
            if sum(ans[j] for j in cur) == -low:
                for j in cur:
                    ans[j] *= -1
                return ans
        return []

    @staticmethod
    def lc_2663(s: str, k: int) -> str:
        """
        url: https://leetcode.cn/problems/lexicographically-smallest-beautiful-string/
        tag: greedy|construction|palindrome_substring|lexicographical_order|reverse_order|brute_force
        """
        n = len(s)
        for i in range(n - 1, -1, -1):
            for x in range(ord(s[i]) - ord("a") + 1, k):
                w = chr(ord("a") + x)
                if (i == 0 or s[i - 1] != w) and not (i >= 2 and w == s[i - 2]):
                    ans = s[:i] + w
                    while len(ans) < n:
                        for y in range(0, k):
                            x = chr(y + ord("a"))
                            if x != ans[-1] and (len(ans) < 2 or ans[-2] != x):
                                ans += x
                                break
                    return ans
        return ""

    @staticmethod
    def lg_p7947(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P7947
        tag: greedy|construction|product_n_sum_k|prime_factorization|brain_teaser
        """
        n, k = ac.read_list_ints()
        ans = []
        for p, c in NumFactor().get_prime_factor(n):
            ans.extend([p] * c)
        if sum(ans) > k:
            ac.st(-1)
        else:
            ans.extend([1] * (k - sum(ans)))
            ac.st(len(ans))
            ac.lst(ans)
        return

    @staticmethod
    def lg_p9101(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9101
        tag: construction|directed_graph|no_circle|classical|number_of_path
        """
        k = ac.read_int()
        ac.st(98)
        ac.lst([33, -1])
        for i in range(2, 34):
            if k & (1 << (i - 2)):
                cur = [i + 32]
            else:
                cur = [-1]
            if i > 2:
                cur.append(i - 1)
            else:
                cur.append(-1)
            ac.lst(cur)
        for i in range(34, 99):
            if i in [34, 66]:
                ac.lst([98, -1])
            elif i == 98:
                ac.lst([-1, -1])
            else:
                ac.lst([i - 1, i - 33 if i >= 67 else i + 31])
        return

    @staticmethod
    def lg_p8976(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8976
        tag: brute_force|construction|classical
        """
        for _ in range(ac.read_int()):
            n, a, b = ac.read_list_ints()
            mid = n // 2 + 1
            if a + b > n * (n + 1) // 2 or ac.max(a, b) > (n // 2) * (mid + n) // 2:
                ac.st(-1)
                continue
            s = n * (n + 1) // 2
            lst = [a, b]
            ans = []
            for i in range(n // 2 + 1):
                if ans:
                    break
                x = n // 2 - i
                for aa, bb in [[0, 1], [1, 0]]:
                    if x:
                        rest = lst[aa] - i * (i + 1) // 2
                        y = math.ceil((rest * 2 / x - x + 1) / 2)
                        y = ac.max(y, i + 1)

                        if y + x - 1 <= n:
                            cur = i * (i + 1) // 2 + x * (y + y + x - 1) // 2
                            if cur >= lst[aa] and s - cur >= lst[bb]:
                                pre = list(range(1, i + 1)) + list(range(y, y + x))
                                post = list(range(i + 1, y)) + list(range(y + x, n + 1))
                                ans = pre + post if aa == 0 else post + pre
                                break
                    else:
                        if n // 2 * (1 + n // 2) // 2 >= a and s - a >= b:
                            ans = list(range(1, n + 1))
                            break
            if not ans:
                ac.st(-1)
            else:
                ac.lst(ans)
        return

    @staticmethod
    def lg_p8910(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8910
        tag: permutation_circle|construction|classical|brain_teaser
        """
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            nex = [0] * (n + 1)
            for i in range(k):
                nex[i] = n - k + i
            for i in range(k, n):
                nex[i] = i - k
            ans = []
            for i in range(n):
                if nex[i] != i:
                    lst = [i]
                    while nex[lst[-1]] != i:
                        lst.append(nex[lst[-1]])
                    m = len(lst)
                    ans.append([n + 1, lst[0] + 1])
                    for x in range(1, m):
                        ans.append([lst[x - 1] + 1, lst[x] + 1])
                    ans.append([lst[m - 1] + 1, n + 1])
                    for x in lst:
                        nex[x] = x
            ac.st(len(ans))
            for a in ans:
                ac.lst(a)
        return

    @staticmethod
    def lg_p8880(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8880
        tag: brain_teaser|construction|odd_even|classical
        """
        n = ac.read_int()
        if n % 2 == 0:
            ac.st(-1)
            return
        nums = ac.read_list_ints()
        ind = {num: i for i, num in enumerate(nums)}
        a = [-1] * n
        b = [-1] * n
        for i in range(n):
            j = (i - 1) % n
            x = (i + j) % n
            a[ind[x]] = i
            b[ind[x]] = j
        ac.lst(a)
        ac.lst(b)
        return

    @staticmethod
    def cf_1823d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1823/problem/D
        tag: greedy|construction|palindrome
        """
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            x = [0] + [x - 3 for x in ac.read_list_ints()]
            c = [0] + [x - 3 for x in ac.read_list_ints()]
            st = "abc"
            ans = ["abc"]
            ind = 0
            for i in range(k):
                dx = x[i + 1] - x[i]
                dc = c[i + 1] - c[i]
                if dx < dc:
                    ac.no()
                    break
                ans.append(chr(ord("d") + i) * dc)
                for _ in range(dx - dc):
                    ans.append(st[ind])
                    ind += 1
                    ind %= 3
            else:
                ac.yes()
                ac.st("".join(ans))
        return

    @staticmethod
    def cf_1722g(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1722/G
        tag: odd_even|xor_property|construction
        """

        # def sum_xor(n):
        #     """xor num of range(0, x+1)"""
        #     if n % 4 == 0:
        #         return n  # (4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0
        #     elif n % 4 == 1:
        #         return 1  # n^(n-1)
        #     elif n % 4 == 2:
        #         return n + 1  # n^(n-1)^(n-2)
        #     return 0  # n^(n-1)^(n-2)^(n-3)

        for _ in range(ac.read_int()):
            n = ac.read_int()  # n >= 3
            if n % 4 == 0:
                ans = list(range(n))
            elif n % 4 == 1:
                ans = [0] + list(range(2, n + 1))
            elif n % 4 == 2:
                ans = list(range(1, n - 1)) + [1 << 20, (1 << 20) | (n - 2)]
            else:
                ans = list(range(1, n + 1))
            ac.lst(ans)
        return

    @staticmethod
    def cf_1005f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1005/problem/F
        tag: construction|shortest_path_spanning_tree|classical|dfs|specific_plan
        """
        n, m, k = ac.read_list_ints()
        edges = [[] for _ in range(n)]
        for i in range(m):
            x, y = ac.read_list_ints_minus_one()
            edges[x].append((y, i))
            edges[y].append((x, i))
        stack = deque([0])
        choose = [0] * m
        visit = [inf] * n
        visit[0] = 1
        while stack:
            x = stack.popleft()
            for y, i in edges[x]:
                if visit[y] == inf:
                    choose[i] = 1
                    stack.append(y)
                    visit[y] = visit[x] + 1
        edges = [[i for y, i in edges[x] if visit[y] + 1 == visit[x]] for x in range(n)]
        del visit
        del choose

        ans = []
        use = [0] * n
        for _ in range(k):
            res = ["0"] * m
            for i in range(1, n):
                res[edges[i][use[i]]] = "1"
            ans.append("".join(res))
            for i in range(1, n):
                if use[i] + 1 < len(edges[i]):
                    use[i] += 1
                    break
                else:
                    use[i] = 0
            else:
                break
        ac.st(len(ans))
        ac.st("\n".join(ans))
        return

    @staticmethod
    def cf_1141g(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1141/G
        tag: construction|dfs|color_method|greedy|classical
        """
        n, k = ac.read_list_ints()
        edges = [[] for _ in range(n)]

        for i in range(n - 1):
            x, y = ac.read_list_ints_minus_one()
            edges[x].append((y, i))
            edges[y].append((x, i))

        if k == n:
            ac.st(1)
            ac.lst([1] * (n - 1))
            return
        degree = [len(x) for x in edges]
        color = sorted(degree, reverse=True)[k]
        ans = [-1] * (n - 1)
        stack = [(0, -1, -1)]
        while stack:
            x, fa, c = stack.pop()
            cur = 1
            for y, i in edges[x]:
                if y != fa:
                    if cur == c:
                        cur += 1
                    if cur > color:
                        cur = 1
                    ans[i] = cur
                    stack.append((y, x, cur))
                    cur += 1
                    if cur > color:
                        cur = 1
        ac.st(color)
        ac.lst(ans)
        return

    @staticmethod
    def abc_251f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc251/tasks/abc251_f
        tag: construction|dfs|bfs|classical
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        for ind in range(m):
            x, y = ac.read_list_ints_minus_one()
            dct[x].append(y)
            dct[y].append(x)

        ind = [0] * n
        ans = []
        stack = [0]
        visit = [0] * n
        visit[0] = 1
        while stack:
            x = stack[-1]
            while ind[x] < len(dct[x]):
                y = dct[x][ind[x]]
                ind[x] += 1
                if not visit[y]:
                    stack.append(y)
                    ans.append([x + 1, y + 1])
                    visit[y] = 1
                    break
            else:
                stack.pop()
        for a in ans:
            ac.lst(a)

        ans = []
        stack = [0]
        visit = [0] * n
        visit[0] = 1
        while stack:
            x = stack.pop()
            for y in dct[x]:
                if not visit[y]:
                    stack.append(y)
                    ans.append([x + 1, y + 1])
                    visit[y] = 1
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def abc_251d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc251/tasks/abc251_d
        tag: construction|brute_force|brain_teaser
        """
        ac.read_int()
        base = list(range(1, 100))
        ans = base[:]
        ans.extend([x * 100 for x in base])
        ans.extend([x * 10000 for x in base])
        ans.append(1000000)
        ac.st(len(ans))
        ac.lst(ans)
        return

    @staticmethod
    def abc_239f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc239/tasks/abc239_f
        tag: implemention|construction|greedy|brain_teaser|union_find
        """
        n, m = ac.read_list_ints()
        degree = ac.read_list_ints()
        uf = UnionFind(n)
        pre = degree[:]
        edges = []
        for _ in range(m):
            a, b = ac.read_list_ints_minus_one()
            if not uf.union(a, b):
                ac.st(-1)
                return
            degree[a] -= 1
            degree[b] -= 1
            edges.append((a, b))
        if min(degree) < 0:
            ac.st(-1)
            return
        group = uf.get_root_part()
        group_degree = []
        group_wait = defaultdict(list)
        for g in group:
            dd = sum(degree[x] for x in group[g])
            if dd:
                heappush(group_degree, (-dd, g))
                for x in group[g]:
                    if degree[x]:
                        heappush(group_wait[g], (-degree[x], x))
        ans = []
        while group_degree:
            if len(group_degree) == 1:
                ac.st(-1)
                return
            d1, g1 = heappop(group_degree)
            d2, g2 = heappop(group_degree)
            d1 = -d1
            d2 = -d2
            d1 -= 1
            d2 -= 1
            dd1, xx1 = heappop(group_wait[g1])
            dd2, xx2 = heappop(group_wait[g2])
            dd1 = -dd1 - 1
            dd2 = -dd2 - 1
            ans.append([xx1 + 1, xx2 + 1])
            if dd1:
                heappush(group_wait[g1], (-dd1, xx1))
            if dd2:
                heappush(group_wait[g2], (-dd2, xx2))

            d = d1 + d2
            if len(group_wait[g1]) > len(group_wait[g2]):
                g = g1
                while group_wait[g2]:
                    heappush(group_wait[g1], heappop(group_wait[g2]))
            else:
                g = g2
                while group_wait[g1]:
                    heappush(group_wait[g2], heappop(group_wait[g1]))
            if d:
                heappush(group_degree, (-d, g))
        if len(ans) == n - m - 1:
            uf = UnionFind(n)
            for a, b in edges:
                pre[a] -= 1
                pre[b] -= 1
                uf.union(a, b)
            for a, b in ans:
                pre[a - 1] -= 1
                pre[b - 1] -= 1
                uf.union(a - 1, b - 1)
            if uf.part != 1 or any(x != 0 for x in pre):
                ac.st(-1)
                return
            for a in ans:
                ac.lst(a)
        else:
            ac.st(-1)
        return

    @staticmethod
    def abc_233f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc233/tasks/abc233_f
        tag: graph|union_find|construction|mst|brain_teaser|classical
        """
        n = ac.read_int()
        p = ac.read_list_ints_minus_one()
        m = ac.read_int()
        dct = [dict() for _ in range(n)]
        uf = UnionFind(n)
        edges = []
        for i in range(m):
            a, b = ac.read_list_ints_minus_one()
            if uf.union(a, b):
                dct[a][b] = i
                dct[b][a] = i
            edges.append((a, b))
        lst = p[:]
        group = uf.get_root_part()
        for g in group:
            vals = [p[x] for x in group[g]]
            ind = group[g]
            vals.sort()
            for i, v in zip(ind, vals):
                lst[i] = v
        if lst != list(range(n)):
            ac.st(-1)
            return
        res = []

        for i in range(n):
            if p[i] != i:
                parent = [-1] * n
                j = p.index(i)
                stack = [(i, -1)]
                visit = [0] * n
                visit[i] = 1
                while stack:
                    x, fa = stack.pop()
                    for y in dct[x]:
                        if y != fa and not visit[y]:
                            stack.append((y, x))
                            parent[y] = x
                            visit[y] = 1
                path = [j]
                while path[-1] != i:
                    path.append(parent[path[-1]])

                m = len(path)
                for x in range(1, m):
                    a, b = path[x - 1], path[x]
                    p[a], p[b] = p[b], p[a]
                    res.append(dct[a][b] + 1)

                for x in range(m - 2, 0, -1):
                    a, b = path[x - 1], path[x]
                    p[a], p[b] = p[b], p[a]
                    res.append(dct[a][b] + 1)
        ac.st(len(res))
        ac.lst(res)
        return

    @staticmethod
    def cc_1(ac=FastIO()):
        """
        url: https://www.codechef.com/problems/ENVPILE
        tag: bfs|construction|classical
        """
        for _ in range(ac.read_int()):
            n, w = ac.read_list_ints()
            parent = [-1] * 5001
            nums = ac.read_list_ints()
            ind = list(range(n))
            ind.sort(key=lambda it: -nums[it])
            stack = [w]
            visit = [0] * 5001
            visit[w] = 1
            while stack:
                nex = []
                for x in stack:
                    for i in ind:
                        if nums[i] <= x:
                            break
                        y = nums[i] - x
                        if not visit[y]:
                            visit[y] = 1
                            nex.append(y)
                            parent[y] = i
                stack = nex[:]
            for x in range(nums[0]):
                if visit[x]:
                    ans = []
                    while x != w:
                        i = parent[x]
                        ans.append(i + 1)
                        x = nums[i] - x
                    ans.reverse()
                    for i in range(n):
                        ans.append(i + 1)
                    ac.st(len(ans))
                    ac.lst(ans)
                    break
            else:
                ac.st(-1)
        return

    @staticmethod
    def cf_1217d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1217/D
        tag: construction|observation|classical
        """
        n, m = ac.read_list_ints()
        dct = [[] for _ in range(n)]
        degree = [0] * n
        edges = []
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            degree[j] += 1
            edges.append((i, j))
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            i = stack.pop()
            for j in dct[i]:
                degree[j] -= 1
                if not degree[j]:
                    stack.append(j)
        if not max(degree):
            ac.st(1)
            ac.lst([1] * m)
        else:
            ac.st(2)
            ans = [1 if i >  j else 2 for i, j in edges]
            ac.lst(ans)
        return

    @staticmethod
    def cf_1364d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1364/D
        tag: dfs_tree|construction|independent_set|union_find|undirected_circle|undirected_local_shortest_circle
        """
        n, m, k = ac.read_list_ints()

        other = []
        dct = [[] for _ in range(n)]
        uf = UnionFind(n)
        for _ in range(m):
            i, j = ac.read_list_ints_minus_one()
            if uf.union(i, j):
                dct[i].append(j)
                dct[j].append(i)
            else:
                other.append((i, j))

        parent = [-1] * n
        stack = [0]
        dis = [0] * n
        while stack:
            x = stack.pop()
            for y in dct[x]:
                if y != parent[x]:
                    parent[y] = x
                    stack.append(y)
                    dis[y] = dis[x] + 1
        if not other:
            odd = [i + 1 for i in range(n) if dis[i] % 2 == 0]
            ac.st(1)
            if len(odd) >= (k + 1) // 2:
                ac.lst(odd[:(k + 1) // 2])
            else:
                even = [i + 1 for i in range(n) if dis[i] % 2]
                ac.lst(even[:(k + 1) // 2])
        else:
            res = []
            for i, j in other:
                if not res or abs(dis[i] - dis[j]) < abs(dis[res[0]] - dis[res[1]]):
                    res = [i, j]
            i, j = res
            pre = [i]
            post = [j]
            while pre[-1] != post[-1]:
                if dis[pre[-1]] < dis[post[-1]]:
                    post.append(parent[post[-1]])
                else:
                    pre.append(parent[pre[-1]])
            pre.extend(post[:-1][::-1])
            if len(pre) <= k:
                ac.st(2)
                ac.st(len(pre))
                ac.lst([x + 1 for x in pre])
            else:
                res = [pre[i] + 1 for i in range(0, len(pre) - 1, 2)]
                ac.st(1)
                ac.lst(res[:(k + 1) // 2])
        return

    @staticmethod
    def cf_1809c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1809/C
        tag: construction|diff_array|reverse_pair
        """
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            pre = list(range(n, -1, -1))
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    if k:
                        k -= 1
                        pre[i], pre[j] = pre[j], pre[i]
                    else:
                        break
            for i in range(n, 0, -1):
                pre[i] -= pre[i - 1]
            ac.lst(pre[1:])
        return
    
    @staticmethod
    def abc_362f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc362/tasks/abc362_f
        tag: construction|greedy|observation
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = ac.read_list_ints_minus_one()
            dct[i].append(j)
            dct[j].append(i)
        sub = [1] * n
        parent = [-1] * n
        stack = [(0, -1)]
        while stack:
            x, fa = stack.pop()
            if x >= 0:
                stack.append((~x, fa))
                for y in dct[x]:
                    if y != fa:
                        stack.append((y, x))
                        parent[y] = x
            else:
                x = ~x
                for y in dct[x]:
                    if y != fa:
                        sub[x] += sub[y]
        root = -1
        for i in range(n):
            lst = [sub[j] for j in dct[i] if j != parent[i]]
            lst.append(n - sum(lst) - 1)
            if max(lst) <= n // 2:
                root = i
                break

        parent = [-1] * n
        stack = [(root, -1)]
        son = [[] for _ in range(n)]
        while stack:
            x, fa = stack.pop()
            for y in dct[x]:
                if y != fa:
                    if parent[x] == -1:
                        parent[y] = y
                    else:
                        parent[y] = parent[x]
                    son[parent[y]].append(y)
                    stack.append((y, x))
        stack = [-len(son[x]) * n - x for x in dct[root]]
        heapify(stack)
        while len(stack) >= 2:
            x = -heappop(stack) % n
            y = -heappop(stack) % n
            ac.lst([son[x].pop() + 1, son[y].pop() + 1])
            if son[x]:
                heappush(stack, -len(son[x]) * n - x)
            if son[y]:
                heappush(stack, -len(son[y]) * n - y)
        if stack:
            x = -heappop(stack) % n
            ac.lst([x + 1, root + 1])
        return

    @staticmethod
    def cf_1635d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1635/problem/D
        tag: fibonacci|brain_teaser|construction
        """
        n, p = ac.read_list_ints()
        mod = 10 ** 9 + 7
        ceil = 2 * 10 ** 5 + 10
        f = [0] * (ceil + 1)
        pre = [0] * (ceil + 1)
        f[1] = f[2] = 1
        pre[1] = 1
        pre[2] = 2
        for i in range(3, ceil + 1):
            f[i] = (f[i - 1] + f[i - 2]) % mod
            pre[i] = (pre[i - 1] + f[i]) % mod

        def check(x):
            while x:
                if x in visit:
                    return False
                if x & 1:
                    x >>= 1
                elif x & 2:
                    break
                else:
                    x >>= 2
            return True

        nums = ac.read_list_ints()
        nums.sort()
        visit = set()
        for num in nums:
            if check(num):
                visit.add(num)
        ans = 0
        for num in visit:
            b = num.bit_length()
            if p - b + 1 > 0:
                ans += pre[p - b + 1]
                ans %= mod
        ac.st(ans)
        return
