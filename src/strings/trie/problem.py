"""
Algorithm：trie|01-trie
Description：string|bit_operation

====================================LeetCode====================================
421（https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/）01-trie
638（https://leetcode.cn/problems/shopping-offers/）trie|memory_search
2416（https://leetcode.cn/problems/sum-of-prefix-scores-of-strings/）prefix|counter
1803（https://leetcode.cn/problems/count-pairs-with-xor-in-a-range/）01-trie|classical
677（https://leetcode.cn/problems/map-sum-pairs/）prefix|counter
2479（https://leetcode.cn/problems/maximum-xor-of-two-non-overlapping-subtrees/）01-trie|maximum_xor
1717（https://leetcode.cn/problems/multi-search-lcci/）ac_auto_machine|counter|trie|reverse_thinking
1707（https://leetcode.cn/problems/maximum-xor-with-an-element-from-array/）sort|offline_query|01-trie
1938（https://leetcode.cn/problems/maximum-genetic-difference-query/）dfs|back_track|01-trie|maximum_xor
1032（https://leetcode.cn/problems/stream-of-characters/description/）trie|classical|reverse_order

=====================================LuoGu======================================
P8306（https://www.luogu.com.cn/problem/P8306）trie
P4551（https://www.luogu.com.cn/problem/P4551）xor
P3864（https://www.luogu.com.cn/problem/P3864）hash|brute_force|trie
P5755（https://www.luogu.com.cn/problem/P5755）trie|counter
P1481（https://www.luogu.com.cn/problem/P1481）trie
P5283（https://www.luogu.com.cn/problem/P5283）trie|kth_xor|heapq|greedy
P2922（https://www.luogu.com.cn/problem/P2922）trie|prefix|counter
P1738（https://www.luogu.com.cn/problem/P1738）trie|counter
P8420（https://www.luogu.com.cn/problem/P8420）trie|greedy

===================================CodeForces===================================
1792D（https://codeforces.com/problemset/problem/1792/D）trie|counter
706D（https://codeforces.com/problemset/problem/706/D）01-trie|maximum_xor
241B（https://codeforces.com/contest/241/problem/B）01-trie|kth_xor|heapq|greedy
665E（https://codeforces.com/contest/665/problem/E）counter|xor_pair
282E（https://codeforces.com/contest/282/problem/E）01-trie|maximum_xor
1902E（https://codeforces.com/contest/1902/problem/E）trie|prefix_count
665E（https://codeforces.com/contest/665/problem/E）01-trie|get_cnt_smaller_xor
817E（https://codeforces.com/contest/817/problem/E）01-trie|get_cnt_smaller_xor

=====================================AcWing=====================================
142（https://www.acwing.com/problem/content/144/）trie|prefix_count
143（https://www.acwing.com/problem/content/145/）maximum_xor|classical
144（https://www.acwing.com/problem/content/description/146/）01-trie|maximum_xor
161（https://www.acwing.com/problem/content/163/）trie

Set Xor-Min（https://judge.yosupo.jp/problem/set_xor_min）template|minimum_xor|classical|update|query
"""
import heapq
import math
from collections import Counter, defaultdict
from typing import List

from src.strings.trie.template import BinaryTrieXor, TrieKeyWordSearchInText, TrieZeroOneXorRange, StringTriePrefix
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_1717(big: str, smalls: List[str]) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/multi-search-lcci/
        tag: ac_auto_machine|counter|trie|reverse_thinking
        """
        # AC自动机类似题目，查询关键词在文本中的出现索引
        trie = TrieKeyWordSearchInText()
        for i, word in enumerate(smalls):
            trie.add_key_word(word, i)

        ans = [[] for _ in smalls]
        n = len(big)
        for i in range(n):
            for j in trie.search_text(big[i:]):
                ans[j].append(i)
        return ans

    @staticmethod
    def lc_677():
        """
        url: https://leetcode.cn/problems/map-sum-pairs/
        tag: prefix|counter
        """
        # 更新与查询给定字符串作为单词键前缀的对应值的和
        class MapSum:
            def __init__(self):
                self.trie = TriePrefixKeyValue()

            def insert(self, key: str, val: int) -> None:
                self.trie.update(key, val)

            def sum(self, prefix: str) -> int:
                return self.trie.query(prefix)

        MapSum()
        return

    @staticmethod
    def lc_1803_1(nums: List[int], low: int, high: int) -> int:
        """
        url: https://leetcode.cn/problems/count-pairs-with-xor-in-a-range/
        tag: 01-trie|classical|inclusion_exclusion
        """
        n = len(nums)
        trie = BinaryTrieXor(max(high, max(nums)), n)
        ans = 0
        for num in nums:
            ans += trie.get_cnt_smaller_xor(num, high)
            ans -= trie.get_cnt_smaller_xor(num, low - 1)
            trie.add(num)
        return ans

    @staticmethod
    def lc_1803_2(nums: List[int], low: int, high: int) -> int:
        """
        url: https://leetcode.cn/problems/count-pairs-with-xor-in-a-range/
        tag: 01-trie|classical|hard
        """
        ans, cnt = 0, Counter(nums)
        high += 1
        while high:
            nxt = Counter()
            for x, c in cnt.items():
                if high & 1:
                    ans += c * cnt[x ^ (high - 1)]
                if low & 1:
                    ans -= c * cnt[x ^ (low - 1)]
                nxt[x >> 1] += c
            cnt = nxt
            low >>= 1
            high >>= 1
        return ans // 2

    @staticmethod
    def cf_706d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/706/D
        tag: 01-trie|maximum_xor
        """
        q = ac.read_int()
        trie = BinaryTrieXor(10 ** 9, q)
        trie.add(0)
        for _ in range(q):
            op, x = ac.read_list_strs()
            if op == "+":
                trie.add(int(x))
            elif op == "-":
                trie.remove(int(x))
            else:
                ac.st(trie.get_maximum_xor(int(x)))
        return

    @staticmethod
    def cf_1902e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1902/problem/E
        tag: trie|prefix_count
        """
        n = ac.read_int()
        words = [ac.read_str() for _ in range(n)]
        trie = StringTriePrefix(sum(len(x) for x in words), n)
        ans = 0
        for i in range(2):
            pre = 0
            for j, word in enumerate(words):
                ans -= trie.count(word[::-1]) * 2
                ans += j * len(word) + pre
                pre += len(word)
                trie.add(word)
            if i == 0:
                words.reverse()
                trie.initial()

        for word in words:
            n = len(word)
            i = n - 1
            j = 0
            while i >= 0:
                if word[i] == word[j]:
                    i -= 1
                    j += 1
                else:
                    break
            ans += 2 * (i + 1)
        ac.st(ans)
        return

    @staticmethod
    def lc_2479(n: int, edges: List[List[int]], values: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-xor-of-two-non-overlapping-subtrees/
        tag: 01-trie|maximum_xor
        """
        # 借助dfs|的顺序01trie查询最大异或数对值
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        # preprocess子树取值之和
        def dfs1(x, fa):
            res = values[x]
            for y in dct[x]:
                if y != fa:
                    dfs1(y, x)
                    res += son[y]
            son[x] = res
            return

        son = [0] * n
        dfs1(0, -1)

        def dfs2(x, fa):
            nonlocal ans
            cur = trie.query_xor_max(son[x])
            ans = ans if ans > cur else cur
            for y in dct[x]:
                if y != fa:
                    dfs2(y, x)
            trie.add(son[x])
            return

        # 根据题意dfs_order和01trie动态维护查询
        trie = TrieZeroOneXorMax(int(math.log2(sum(values)) + 2))
        ans = 0
        dfs2(0, -1)
        return ans

    @staticmethod
    def lc_2416(words: List[str]) -> List[int]:
        """
        url: https://leetcode.cn/problems/sum-of-prefix-scores-of-strings/
        tag: prefix|counter
        """
        trie = TriePrefixCount()
        for word in words:
            trie.update(word)
        return [trie.query(word) for word in words]

    @staticmethod
    def lg_p1481(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1481
        tag: trie
        """
        # trie最长词链
        n = ac.read_int()

        dct = dict()
        ans = 0
        for _ in range(n):
            cur = dct
            x = 0
            for w in ac.read_str():
                if "isEnd" in cur:
                    x += 1
                if w not in cur:
                    cur[w] = dict()
                cur = cur[w]
            cur["isEnd"] = 1
            x += 1
            ans = ans if ans > x else x
        ac.st(ans)
        return

    @staticmethod
    def lg_p4551(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4551
        tag: xor
        """
        # 树中异或值最长的路径
        n = ac.read_int()
        trie = TrieZeroOneXorMax(32)

        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, k = ac.read_list_ints()
            dct[i - 1][j - 1] = k
            dct[j - 1][i - 1] = k

        stack = [[0, -1, 0]]
        ans = 0
        while stack:
            i, fa, pre = stack.pop()
            ans = ac.max(ans, trie.query_xor_max(pre))
            ans = ac.max(ans, pre)
            trie.add(pre)
            for j in dct[i]:
                if j != fa:
                    stack.append([j, i, pre ^ dct[i][j]])
        ac.st(ans)
        return

    @staticmethod
    def lg_p5283(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5283
        tag: trie|kth_xor|heapq|greedy
        """
        mod = 10 ** 9 + 7
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        trie = BinaryTrieXor(max(nums), n)
        for i, num in enumerate(nums):
            trie.add(num)
        stack = [(-trie.get_kth_maximum_xor(nums[i], 1), i, 1) for i in range(n)]
        heapq.heapify(stack)
        ans = 0
        for _ in range(2 * k):
            num, i, c = heapq.heappop(stack)
            ans -= num
            if c + 1 <= n:
                heapq.heappush(stack, (-trie.get_kth_maximum_xor(nums[i], c + 1), i, c + 1))
        ac.st((ans // 2) % mod)
        return

    @staticmethod
    def cf_241b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/241/problem/B
        tag: 01-trie|kth_xor|heapq|greedy
        """
        # 数组中最大的 k 组异或对
        mod = 10 ** 9 + 7
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        trie = TrieZeroOneXorMaxKth(len(bin(max(nums))))
        for i, num in enumerate(nums):
            trie.add(num)
        stack = [(-trie.query_xor_kth_max(nums[i], 1), i, 1) for i in range(n)]
        heapq.heapify(stack)
        ans = 0
        for _ in range(2 * k):
            num, i, c = heapq.heappop(stack)
            ans -= num
            if c + 1 <= n:
                heapq.heappush(stack, (-trie.query_xor_kth_max(nums[i], c + 1), i, c + 1))
        ac.st((ans // 2) % mod)
        return

    @staticmethod
    def ac_143(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/145/
        tag: maximum_xor|classical
        """
        # 最大异或对
        ac.read_int()
        ans = 0
        trie = TrieZeroOneXorMax(32)
        for num in ac.read_list_ints():
            ans = ac.max(ans, trie.query_xor_max(num))
            trie.add(num)
        ac.st(ans)
        return

    @staticmethod
    def ac_144(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/146/
        tag: 01-trie|maximum_xor
        """
        # 01-trie树中最长异或路径
        n = ac.read_int()
        dct = [dict() for _ in range(n)]
        for _ in range(n - 1):
            i, j, w = ac.read_list_ints()
            dct[i][j] = w
            dct[j][i] = w

        ans = 0
        trie = BinaryTrieXor(32)

        stack = [[0, -1, 0]]
        ceil = (1 << 31) - 1
        while stack:
            i, fa, val = stack.pop()
            ans = max(ans, trie.max_xor(val))
            if ans == ceil:
                break
            trie.add(val)
            for j in dct[i]:
                if j != fa:
                    stack.append([j, i, val ^ dct[i][j]])
        ac.st(ans)
        return

    @staticmethod
    def ac_161(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/163/
        tag: trie
        """
        # O(n)trie判断是否存在单词前缀包含
        for _ in range(ac.read_int()):
            n = ac.read_int()
            dct = dict()
            ans = True
            for _ in range(n):
                s = ac.read_str()
                if not ans:
                    continue
                cur = dct
                flag = True
                for w in s:
                    if w not in cur:
                        cur[w] = dict()
                        flag = False
                    cur = cur[w]
                    if "cnt" in cur:
                        ans = False
                        break
                if flag:
                    ans = False
                cur["cnt"] = 1
            ac.st("YES" if ans else "NO")
            del dct
        return

    @staticmethod
    def lg_p2922(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2922
        tag: trie|prefix|counter
        """
        # trie前缀匹配
        m, n = ac.read_list_ints()
        dct = dict()
        for _ in range(m):
            b = ac.read_list_ints()
            cur = dct
            for num in b[1:]:
                if num not in cur:
                    cur[num] = dict()
                cur = cur[num]
                cur["mid"] = cur.get("mid", 0) + 1
            cur["cnt"] = cur.get("cnt", 0) + 1

        for _ in range(n):
            ans = 0
            cur = dct
            for num in ac.read_list_ints()[1:]:
                if num not in cur:
                    break
                cur = cur[num]
                ans += cur.get("cnt", 0)
            else:
                ans += cur.get("mid", 0) - cur.get("cnt", 0)
            ac.st(ans)
        return

    @staticmethod
    def lg_p1738(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1738
        tag: trie|counter
        """
        # 动态维护trie键个数
        n = ac.read_int()
        dct = dict()
        ans = 0
        for _ in range(n):
            s = ac.read_str().split("/")[1:]
            cur = dct
            for w in s:
                if w not in cur:
                    ans += 1
                    cur[w] = dict()
                cur = cur[w]
            ac.st(ans)
        return

    @staticmethod
    def lg_p8420(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8420
        tag: trie|greedy
        """
        # triegreedy匹配
        n, m, length = ac.read_list_ints()

        # 后缀 0 1 匹配的代价和
        cnt = [0] * length
        for _ in range(n):
            s = ac.read_str()
            for i in range(length):
                cnt[i] += 1 if s[i] == "1" else 0
        post = [0] * (length + 1)
        for i in range(length - 1, -1, -1):
            post[i] = post[i + 1] + ac.min(n - cnt[i], cnt[i])

        # 禁用词构建字典
        dct = dict()
        for _ in range(m):
            s = ac.read_str()
            cur = dct
            for w in s:
                if w not in cur:
                    cur[w] = dict()
                cur = cur[w]

        def dfs(x, cur_dct, p):
            nonlocal ans
            if x == length:
                return
            if "1" in cur_dct:
                dfs(x + 1, cur_dct["1"], p + n - cnt[x])
            else:
                # 当前键为空时greedy匹配
                ans = ac.min(ans, p + n - cnt[x] + post[x + 1])
            if "0" in cur_dct:
                dfs(x + 1, cur_dct["0"], p + cnt[x])
            else:
                ans = ac.min(ans, p + cnt[x] + post[x + 1])
            return

        ans = inf
        dfs(0, dct, 0)
        ac.st(ans)
        return

    @staticmethod
    def lc_1707(nums: List[int], queries: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/maximum-xor-with-an-element-from-array/
        tag: sort|offline_query|01-trie
        """
        # sorting后offline_query并 01 Trie求解最大异或值
        n = len(nums)
        nums.sort()

        # 添|pointer
        m = len(queries)
        for i in range(m):
            queries[i].append(i)
        queries.sort(key=lambda it: it[1])

        # pointeroffline_query
        trie = TrieZeroOneXorMax(32)
        ans = [-1] * m
        j = 0
        for x, m, i in queries:
            while j < n and nums[j] <= m:
                trie.add(nums[j])
                j += 1
            if trie.dct:
                ans[i] = trie.query_xor_max(x)
            else:
                ans[i] = -1
        return ans

    @staticmethod
    def cf_665e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/665/problem/E
        tag: counter|xor_pair
        """
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        for i in range(1, n):
            nums[i] ^= nums[i - 1]
        cnt = {0: 1}
        for num in nums:
            cnt[num] = cnt.get(num, 0) + 1
        # 统计范围内的异或对数目
        ans = 0
        del nums
        high = 1 << 30
        low = k
        while high:
            nxt = dict()
            for x in cnt:
                c = cnt[x]
                if high & 1:
                    ans += c * cnt.get(x ^ (high - 1), 0)
                if low & 1:
                    ans -= c * cnt.get(x ^ (low - 1), 0)
                nxt[x >> 1] = nxt.get(x >> 1, 0) + c
            cnt = nxt
            low >>= 1
            high >>= 1
        ac.st(ans // 2)
        return

    @staticmethod
    def lib_check_1(ac=FastIO()):
        """template of set xor min"""
        bt = BinaryTrieXor(32)
        dct = set()
        for _ in range(ac.read_int()):
            op, x = ac.read_list_ints()
            if op == 0:
                if x not in dct:
                    dct.add(x)
                    bt.add(x)
            elif op == 1:
                if x in dct:
                    dct.discard(x)
                    bt.remove(x)
            else:
                ac.st(bt.min_xor(x))
        return

    @staticmethod
    def lc_421(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/
        tag: 01-trie
        """
        # 求解数组最大的异或对，题，有更快解法
        trie = TrieZeroOneXorMax(32)
        ans = 0
        for num in nums:
            cur = trie.query_xor_max(num)
            ans = ans if ans > cur else cur
            trie.add(num)
        return ans

    @staticmethod
    def lc_421_2(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/
        tag: 01-trie
        """
        # 更快解法
        res = 0
        mask = 0
        max_len = len(bin(max(nums))) - 2
        # 最大长度不会超过最大值，异或的特性
        for i in range(max_len - 1, -1, -1):
            cur = 1 << i
            mask = mask | cur
            res |= cur
            d = {}
            find = 0
            for num in nums:
                d[num & mask] = 1
                if (num & mask) ^ res in d:
                    find = 1
                    break
            if not find:
                res ^= 1 << i
        return res

    @staticmethod
    def cf_282e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/282/problem/E
        tag: 01-trie|maximum_xor
        """
        # 维护和查询最大异或数值对
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = pre = 0
        trie = BinaryTrieXor(40)
        trie.add(0)
        for i in range(n):
            pre ^= nums[i]
            trie.add(pre)
            ans = ac.max(ans, pre)

        pre = 0
        for i in range(n - 1, -1, -1):
            pre ^= nums[i]
            ans = ac.max(ans, trie.max_xor(pre))
        ac.st(ans)
        return

    @staticmethod
    def lc_1938(parents: List[int], queries: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/maximum-genetic-difference-query/
        tag: dfs|back_track|01-trie|maximum_xor
        """
        # dfs|back_track|结合01-trieoffline_query最大异或值对
        n = len(parents)
        x = -1
        dct = [[] for _ in range(n)]
        for i in range(n):
            if parents[i] == -1:
                x = i
            else:
                dct[parents[i]].append(i)

        query = [dict() for _ in range(n)]
        for node, val in queries:
            query[node][val] = 0
        trie = BinaryTrieXor(20)

        def dfs(a):
            trie.add(a)
            for v in query[a]:
                query[a][v] = trie.max_xor(v)
            for b in dct[a]:
                dfs(b)
            trie.remove(a)
            return

        dfs(x)
        return [query[node][val] for node, val in queries]

    @staticmethod
    def lc_1938_2(parents: List[int], queries: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/maximum-genetic-difference-query/
        tag: dfs|back_track|01-trie|maximum_xor
        """
        # dfs|back_track|结合01-trieoffline_query最大异或值对
        n = len(parents)
        dct = [[] for _ in range(n)]
        root = -1
        for i in range(n):
            if parents[i] == -1:
                root = i
            else:
                dct[parents[i]].append(i)

        ind = [defaultdict(dict) for _ in range(n)]
        for node, val in queries:
            ind[node][val] = 0

        def dfs(i):
            tree.update(i)
            for v in ind[i]:
                ind[i][v] = tree.query(v)
            for j in dct[i]:
                dfs(j)
            tree.delete(i)
            return

        tree = TrieBit()
        dfs(root)
        return [ind[node][v] for node, v in queries]

    @staticmethod
    def cf_665e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/665/problem/E
        tag: 01-trie|get_cnt_smaller_xor
        """
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        pre = 0
        trie = BinaryTrieXor(reduce(or_, nums), n + 1)
        trie.add(0)
        ans = n * (n + 1) // 2
        for num in nums:
            pre ^= num
            ans -= trie.get_cnt_smaller_xor(pre, k - 1)
            trie.add(pre)
        ac.st(ans)
        return