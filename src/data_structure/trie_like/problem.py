"""
Algorithm：trie_like|binary_trie
Description：string|bit_operation

====================================LeetCode====================================
421（https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/）binary_trie
638（https://leetcode.cn/problems/shopping-offers/）trie_like|memory_search
2416（https://leetcode.cn/problems/sum-of-prefix-scores-of-strings/）prefix|counter
1803（https://leetcode.cn/problems/count-pairs-with-xor-in-a-range/）binary_trie|classical
677（https://leetcode.cn/problems/map-sum-pairs/）prefix|counter
2479（https://leetcode.cn/problems/maximum-xor-of-two-non-overlapping-subtrees/）binary_trie|maximum_xor
1717（https://leetcode.cn/problems/multi-search-lcci/）automaton|counter|trie_like|reverse_thinking
1707（https://leetcode.cn/problems/maximum-xor-with-an-element-from-array/）sort|offline_query|binary_trie
1938（https://leetcode.cn/problems/maximum-genetic-difference-query/）dfs|back_track|binary_trie|maximum_xor
1032（https://leetcode.cn/problems/stream-of-characters/description/）trie_like|classical|reverse_order
1554（https://leetcode.cn/problems/strings-differ-by-one-character/）string_hash|trie
2935（https://leetcode.cn/problems/maximum-strong-pair-xor-ii/description/）binary_trie|hash|bit_operation

=====================================LuoGu======================================
P8306（https://www.luogu.com.cn/problem/P8306）trie_like
P4551（https://www.luogu.com.cn/problem/P4551）xor
P3864（https://www.luogu.com.cn/problem/P3864）hash|brute_force|trie_like
P5755（https://www.luogu.com.cn/problem/P5755）trie_like|counter
P1481（https://www.luogu.com.cn/problem/P1481）trie_like
P5283（https://www.luogu.com.cn/problem/P5283）trie_like|kth_xor|heapq|greedy
P2922（https://www.luogu.com.cn/problem/P2922）trie_like|prefix|counter
P1738（https://www.luogu.com.cn/problem/P1738）trie_like|counter
P8420（https://www.luogu.com.cn/problem/P8420）trie_like|greedy
P4735（https://www.luogu.com.cn/problem/P4735）

===================================CodeForces===================================
1792D（https://codeforces.com/problemset/problem/1792/D）trie_like|counter
706D（https://codeforces.com/problemset/problem/706/D）binary_trie|maximum_xor
241B（https://codeforces.com/contest/241/problem/B）binary_trie|kth_xor|heapq|greedy
665E（https://codeforces.com/contest/665/problem/E）counter|xor_pair
282E（https://codeforces.com/contest/282/problem/E）binary_trie|maximum_xor
1902E（https://codeforces.com/contest/1902/problem/E）trie_like|prefix_count
665E（https://codeforces.com/contest/665/problem/E）binary_trie|get_cnt_smaller_xor
817E（https://codeforces.com/contest/817/problem/E）binary_trie|get_cnt_smaller_xor
1777F（https://codeforces.com/problemset/problem/1777/F）
1446C（https://codeforces.com/contest/1446/problem/C）
923C（https://codeforces.com/problemset/problem/923/C）binary_trie|greedy
1055F（https://codeforces.com/problemset/problem/1055/F）binary_trie|get_cnt_smaller_xor
1720D2（https://codeforces.com/contest/1720/problem/D2）
1849F（https://codeforces.com/problemset/problem/1849/F）
888G（https://codeforces.com/problemset/problem/888/G）

=====================================AcWing=====================================
144（https://www.acwing.com/problem/content/144/）trie_like|prefix_count
145（https://www.acwing.com/problem/content/145/）maximum_xor|classical
146（https://www.acwing.com/problem/content/description/146/）binary_trie|maximum_xor
163（https://www.acwing.com/problem/content/163/）trie_like
258（https://www.acwing.com/problem/content/258/）

=====================================LibraryChecker=====================================
1（https://judge.yosupo.jp/problem/set_xor_min）template|minimum_xor|classical|update|query

"""
import heapq
from collections import Counter, defaultdict
from functools import reduce
from operator import or_
from typing import List

from src.data_structure.trie_like.template import BinaryTrieXor, StringTriePrefix, StringTrieSearch, \
    BinaryTrieXorLimited
from src.utils.fast_io import FastIO
from src.utils.fast_io import inf


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_1717(big: str, smalls: List[str]) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/multi-search-lcci/
        tag: automaton|counter|trie_like|reverse_thinking
        """
        n = len(smalls)
        sts = StringTrieSearch(sum(len(x) for x in smalls) + 1, n)
        for i, word in enumerate(smalls):
            if word:
                sts.add([ord(w) - ord("a") for w in word], i + 1)

        ans = [[] for _ in range(n)]
        for i in range(len(big)):
            for j in sts.search([ord(w) - ord("a") for w in big[i:]]):
                ans[j - 1].append(i)
        return ans

    @staticmethod
    def lc_677():
        """
        url: https://leetcode.cn/problems/map-sum-pairs/
        tag: prefix|counter
        """

        class MapSum:

            def __init__(self):
                self.trie = StringTriePrefix(50 * 50, 5 * 10 ** 4)
                self.dct = defaultdict(int)

            def insert(self, key: str, val: int) -> None:
                self.trie.add([ord(w) - ord("a") for w in key], val - self.dct[key])
                self.dct[key] = val
                return

            def sum(self, prefix: str) -> int:
                return self.trie.count_end([ord(w) - ord("a") for w in prefix])

        return MapSum()

    @staticmethod
    def lc_1803_1(nums: List[int], low: int, high: int) -> int:
        """
        url: https://leetcode.cn/problems/count-pairs-with-xor-in-a-range/
        tag: binary_trie|classical|inclusion_exclusion
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
        tag: binary_trie|classical|hard
        """
        ans, cnt = 0, Counter(nums)
        high += 1
        while high:
            nxt = Counter()
            for x, c in cnt.items():
                if high & 1:
                    ans += c * cnt[x ^ (high ^ 1)]
                if low & 1:
                    ans -= c * cnt[x ^ (low ^ 1)]
                nxt[x >> 1] += c
            cnt = nxt
            low >>= 1
            high >>= 1
        return ans // 2

    @staticmethod
    def cf_706d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/706/D
        tag: binary_trie|maximum_xor
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
        tag: trie_like|prefix_count
        """
        n = ac.read_int()
        words = [ac.read_str() for _ in range(n)]
        trie = StringTriePrefix(sum(len(x) for x in words), n)
        ans = 0
        for i in range(2):
            pre = 0
            for j, word in enumerate(words):
                ans -= trie.count([ord(w) - ord("a") for w in word[::-1]]) * 2
                ans += j * len(word) + pre
                pre += len(word)
                trie.add([ord(w) - ord("a") for w in word])
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
        tag: binary_trie|maximum_xor|classical|dfs_order|implemention
        """
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        stack.append((j, i))
            else:
                i = ~i
                for j in dct[i]:
                    if j != fa:
                        values[i] += values[j]
        trie = BinaryTrieXor(values[0], n)
        ans = 0
        stack = [(0, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                ans = max(ans, trie.get_maximum_xor(values[i]))
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        stack.append((j, i))
            else:
                i = ~i
                trie.add(values[i])
        return ans

    @staticmethod
    def lc_2416(words: List[str]) -> List[int]:
        """
        url: https://leetcode.cn/problems/sum-of-prefix-scores-of-strings/
        tag: prefix|counter
        """
        trie = StringTriePrefix(sum(len(x) for x in words), len(words))
        for word in words:
            trie.add([ord(w) - ord("a") for w in word])
        return [trie.count([ord(w) - ord("a") for w in word]) for word in words]

    @staticmethod
    def lg_p1481(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1481
        tag: trie_like
        """
        n = ac.read_int()
        words = [ac.read_str() for _ in range(n)]
        trie = StringTrieSearch(sum(len(x) for x in words), n)
        ans = 0
        for i in range(n):
            ans = ac.max(ans, trie.add_cnt([ord(w) - ord("a") for w in words[i]], i + 1))
        ac.st(ans)
        return

    @staticmethod
    def lg_p4551_1(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4551
        tag: get_maximum_xor|binary_trie|hash|implemention
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v, w = ac.read_list_ints_minus_one()
            dct[u].append((v, w + 1))
        trie = BinaryTrieXor((1 << 31) - 1, n)
        ans = 0
        stack = [(0, 0)]
        while stack:
            i, v = stack.pop()
            ans = ac.max(ans, trie.get_maximum_xor(v))
            trie.add(v)
            for j, w in dct[i]:
                stack.append((j, w ^ v))
        ac.st(ans)
        return

    @staticmethod
    def lg_p4551_2(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4551
        tag: get_maximum_xor|binary_trie|hash|implemention
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v, w = ac.read_list_ints_minus_one()
            dct[u].append((v, w + 1))

        xor = [0] * n
        stack = [(0, 0)]
        while stack:
            i, v = stack.pop()
            xor[i] = v
            for j, w in dct[i]:
                stack.append((j, w ^ v))
        ans = 0
        for i in range(30, -1, -1):
            pre = set()
            cur = (ans >> i) | 1
            for num in xor:
                if cur ^ (num >> i) in pre:
                    ans |= (1 << i)
                    break
                pre.add(num >> i)
        ac.st(ans)
        return

    @staticmethod
    def lg_p5283(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5283
        tag: trie_like|kth_xor|heapq|greedy
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
        tag: binary_trie|kth_xor|heapq|greedy
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
    def ac_145_1(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/145/
        tag: maximum_xor|classical
        """
        n = ac.read_int()
        ans = 0
        trie = BinaryTrieXor((1 << 31) - 1, n)
        for num in ac.read_list_ints():
            ans = ac.max(ans, trie.get_maximum_xor(num))
            trie.add(num)
        ac.st(ans)
        return

    @staticmethod
    def ac_145_2(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/145/
        tag: maximum_xor|classical
        """
        ac.read_int()
        nums = ac.read_list_ints()
        ans = 0
        for i in range(30, -1, -1):
            pre = set()
            cur = (ans >> i) | 1
            for num in nums:
                if cur ^ (num >> i) in pre:
                    ans |= (1 << i)
                    break
                pre.add(num >> i)
        ac.st(ans)
        return

    @staticmethod
    def ac_146_1(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/146/
        tag: binary_trie|maximum_xor|hash
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v, w = ac.read_list_ints()
            dct[u].append((v, w))
            dct[v].append((u, w))
        trie = BinaryTrieXor((1 << 31) - 1, n)
        ans = 0
        stack = [(0, -1, 0)]
        while stack:
            i, fa, v = stack.pop()
            ans = ac.max(ans, trie.get_maximum_xor(v))
            trie.add(v)
            for j, w in dct[i]:
                if j != fa:
                    stack.append((j, i, w ^ v))
        ac.st(ans)
        return

    @staticmethod
    def ac_146_2(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/146/
        tag: binary_trie|maximum_xor|hash
        """
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            u, v, w = ac.read_list_ints()
            dct[u].append((v, w))
            dct[v].append((u, w))

        xor = [0] * n
        stack = [(0, -1, 0)]
        while stack:
            i, fa, v = stack.pop()
            xor[i] = v
            for j, w in dct[i]:
                if j != fa:
                    stack.append((j, i, w ^ v))

        ans = 0
        for i in range(30, -1, -1):
            pre = set()
            cur = (ans >> i) | 1
            for num in xor:
                if cur ^ (num >> i) in pre:
                    ans |= (1 << i)
                    break
                pre.add(num >> i)
        ac.st(ans)
        return

    @staticmethod
    def ac_144(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/144/
        tag: trie_like|prefix_count
        """
        n, m = ac.read_list_ints()
        trie = StringTriePrefix(10 ** 6, n)
        for i in range(n):
            trie.add_end([ord(w) - ord("a") for w in ac.read_str()], 1)
        for _ in range(m):
            ac.st(trie.count_pre_end([ord(w) - ord("a") for w in ac.read_str()]))
        return

    @staticmethod
    def ac_163(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/163/
        tag: trie_like
        """

        class StringTriePrefixSP:
            def __init__(self, most_word, word_cnt, string_state=26):  # prefix count
                assert most_word >= 1
                assert word_cnt >= 1
                self.string_state = string_state
                self.cnt_bit = word_cnt.bit_length()
                self.node_cnt = most_word * self.string_state
                self.son_and_cnt = [0] * (self.node_cnt + 1)
                self.end = [0] * (self.node_cnt + 1)
                self.ind = 0
                self.mask = (1 << self.cnt_bit) - 1

            def add_query(self, word):
                cur = 0
                res = True
                for w in word:
                    bit = ord(w) - ord("a")
                    if not self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit:
                        self.ind += 1
                        self.son_and_cnt[bit + cur * self.string_state] |= self.ind << self.cnt_bit
                    cur = self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit
                    if self.end[cur]:
                        res = False
                    self.son_and_cnt[cur] += 1
                if self.son_and_cnt[cur] & self.mask > 1:
                    res = False
                self.end[cur] = 1
                return res

        for _ in range(ac.read_int()):
            n = ac.read_int()
            trie = StringTriePrefixSP(10 ** 5, n)
            ans = "YES"
            for _ in range(n):
                s = ac.read_str()
                if ans == "YES":
                    if not trie.add_query(s):
                        ans = "NO"
            ac.st(ans)
        return

    @staticmethod
    def lg_p2922(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2922
        tag: trie_like|prefix|counter|classical|hard|inclusion_exclusion
        """

        class StringTriePrefixSP:
            def __init__(self, most_word, word_cnt, string_state=2):  # prefix count
                assert most_word >= 1
                assert word_cnt >= 1
                self.string_state = string_state
                self.cnt_bit = word_cnt.bit_length()
                self.node_cnt = most_word * self.string_state
                self.son_and_cnt = [0] * (self.node_cnt + 1)
                self.end = [0] * (self.node_cnt + 1)
                self.ind = 0
                self.mask = (1 << self.cnt_bit) - 1

            def add(self, word):
                cur = 0
                for w in word:
                    bit = int(w)
                    if not self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit:
                        self.ind += 1
                        self.son_and_cnt[bit + cur * self.string_state] |= self.ind << self.cnt_bit
                    cur = self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit
                    self.son_and_cnt[cur] += 1
                self.end[cur] += 1
                return

            def count_prefix(self, word):
                cur = res = 0
                for w in word:
                    bit = int(w)
                    if not self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit:
                        return res
                    cur = self.son_and_cnt[bit + cur * self.string_state] >> self.cnt_bit
                    res += self.end[cur]
                res += (self.son_and_cnt[cur] & self.mask) - self.end[cur]
                return res

        m, n = ac.read_list_ints()
        trie = StringTriePrefixSP(5 * 10 ** 5, m)
        for _ in range(m):
            b = ac.read_list_strs()
            trie.add("".join(b[1:]))

        for _ in range(n):
            b = ac.read_list_strs()
            ac.st(trie.count_prefix("".join(b[1:])))
        return

    @staticmethod
    def lg_p1738(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1738
        tag: trie_like|counter
        """
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
        tag: trie_like|greedy
        """
        n, m, length = ac.read_list_ints()
        cnt = [0] * length
        for _ in range(n):
            s = ac.read_str()
            for i in range(length):
                cnt[i] += 1 if s[i] == "1" else 0

        cost = [0] * (length + 1)
        for i in range(length - 1, -1, -1):
            cost[i] = cost[i + 1] + ac.min(n - cnt[i], cnt[i])

        trie = StringTrieSearch(m * length, m, 2)
        for i in range(m):
            word = ac.read_str()
            trie.add_bin([ord(w) - ord("a") for w in word], i + 1)

        ans = inf
        stack = [(0, 0, 0)]
        while stack:
            x, cur, p = stack.pop()
            if x == length:
                continue

            one = trie.son_and_ind[1 + cur * trie.string_state] >> trie.cnt_bit
            if one:
                stack.append((x + 1, one, p + n - cnt[x]))
            else:
                ans = ac.min(ans, p + n - cnt[x] + cost[x + 1])

            zero = trie.son_and_ind[cur * trie.string_state] >> trie.cnt_bit
            if zero:
                stack.append((x + 1, zero, p + cnt[x]))
            else:
                ans = ac.min(ans, p + cnt[x] + cost[x + 1])
        ac.st(ans)
        return

    @staticmethod
    def lc_1707_1(nums: List[int], queries: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/maximum-xor-with-an-element-from-array/
        tag: sort|offline_query|binary_trie
        """
        n = len(nums)
        nums.sort()

        m = len(queries)
        for i in range(m):
            queries[i].append(i)
        queries.sort(key=lambda it: it[1])

        trie = BinaryTrieXor(10 ** 9, n)
        ans = [-1] * m
        j = 0
        for x, m, i in queries:
            while j < n and nums[j] <= m:
                trie.add(nums[j])
                j += 1
            if trie.son_and_cnt[0] & trie.mask:
                ans[i] = trie.get_maximum_xor(x)
            else:
                ans[i] = -1
        return ans

    @staticmethod
    def lc_1707_2(nums: List[int], queries: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/maximum-xor-with-an-element-from-array/
        tag: sort|offline_query|binary_trie
        """
        n = len(nums)
        trie = BinaryTrieXorLimited(10 ** 9, n)
        for num in nums:
            trie.add(num)
        ans = [trie.get_maximum_xor_limited(x, m) for x, m in queries]
        return ans

    @staticmethod
    def cf_665e_1(ac=FastIO()):
        """
        url: https://codeforces.com/contest/665/problem/E
        tag: binary_trie|get_cnt_smaller_xor
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

    @staticmethod
    def cf_665e_2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/665/problem/E
        tag: counter|xor_pair
        """
        n, k = ac.read_list_ints()  # TLE
        nums = ac.read_list_ints()
        for i in range(1, n):
            nums[i] ^= nums[i - 1]
        cnt = {0: 1}
        for num in nums:
            cnt[num] = cnt.get(num, 0) + 1
        ans = 0
        del nums
        high = 1 << 30
        low = k
        while high:
            nxt = dict()
            for x in cnt:
                c = cnt[x]
                if high & 1:
                    ans += c * cnt.get(x ^ (high ^ 1), 0)
                if low & 1:
                    ans -= c * cnt.get(x ^ (low ^ 1), 0)
                nxt[x >> 1] = nxt.get(x >> 1, 0) + c
            cnt = nxt
            low >>= 1
            high >>= 1
        ac.st(ans // 2)
        return

    @staticmethod
    def lib_check_1(ac=FastIO()):
        """
        url: https://judge.yosupo.jp/problem/set_xor_min
        tag: template|minimum_xor|classical|update|query
        """
        q = ac.read_int()
        trie = BinaryTrieXor((1 << 30) - 1, q)
        dct = set()
        for _ in range(q):
            op, x = ac.read_list_ints()
            if op == 0:
                if x not in dct:
                    dct.add(x)
                    trie.add(x)
            elif op == 1:
                if x in dct:
                    dct.discard(x)
                    trie.remove(x)
            else:
                ac.st(trie.get_minimum_xor(x))
        return

    @staticmethod
    def lc_421_1(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/
        tag: binary_trie|hash
        """
        trie = BinaryTrieXor(max(nums) + 1, len(nums))
        ans = 0
        for num in nums:
            cur = trie.get_maximum_xor(num)
            ans = ans if ans > cur else cur
            trie.add(num)
        return ans

    @staticmethod
    def lc_421_2(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-xor-of-two-numbers-in-an-array/
        tag: binary_trie|hash
        """
        m = max(nums).bit_length() - 1  # faster!
        ans = 0
        for i in range(m, -1, -1):
            cur = (ans >> i) | 1
            pre = set()
            for num in nums:
                if cur ^ (num >> i) in pre:
                    ans |= 1 << i
                    break
                pre.add(num >> i)
        return ans

    @staticmethod
    def cf_282e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/282/problem/E
        tag: binary_trie|maximum_xor
        """

        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = pre = 0
        trie = BinaryTrieXor(10 ** 12, n + 1)
        trie.add(0)
        for i in range(n):
            pre ^= nums[i]
            trie.add(pre)
            ans = ac.max(ans, pre)

        pre = 0
        for i in range(n - 1, -1, -1):
            pre ^= nums[i]
            ans = ac.max(ans, trie.get_maximum_xor(pre))
        ac.st(ans)
        return

    @staticmethod
    def lc_1938(parents: List[int], queries: List[List[int]]) -> List[int]:
        """
        url: https://leetcode.cn/problems/maximum-genetic-difference-query/
        tag: dfs|back_track|binary_trie|maximum_xor
        """
        n = len(parents)
        root = -1
        dct = [[] for _ in range(n)]
        for i in range(n):
            if parents[i] == -1:
                root = i
            else:
                dct[parents[i]].append(i)
        ceil = n
        query = [dict() for _ in range(n)]
        for node, val in queries:
            query[node][val] = 0
            ceil = max(ceil, val)
        trie = BinaryTrieXor(ceil, n)
        stack = [root]
        while stack:
            i = stack.pop()
            if i >= 0:
                trie.add(i)
                stack.append(~i)
                for w in query[i]:
                    query[i][w] = trie.get_maximum_xor(w)
                for j in dct[i]:
                    stack.append(j)
            else:
                trie.remove(~i)
        ans = [query[node][val] for node, val in queries]
        return ans

    @staticmethod
    def lc_1554(words: List[str]) -> bool:
        """
        url: https://leetcode.cn/problems/strings-differ-by-one-character/
        tag: string_hash|trie
        """
        n, m = len(words), len(words[0])
        trie = StringTrieSearch(m * n, n, 26)
        for i, word in enumerate(words):
            lst = [ord(w) - ord("a") for w in word]
            if trie.search_for_one_difference(lst):
                return True
            trie.add_int(lst, i + 1)
        return False

    @staticmethod
    def lc_2935_1(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-strong-pair-xor-ii/description/
        tag: binary_trie|hash|bit_operation
        """
        ceil = max(nums)
        n = len(nums)
        nums.sort()
        trie = BinaryTrieXor(ceil, n)
        ans = j = 0
        for i in range(n):
            while j < n and nums[j] - nums[i] <= nums[i]:
                cur = trie.get_maximum_xor(nums[j])
                trie.add(nums[j], 1)
                if cur > ans:
                    ans = cur
                j += 1
            trie.remove(nums[i], 1)
        return ans

    @staticmethod
    def lc_2935_2(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximum-strong-pair-xor-ii/description/
        tag: binary_trie|hash|bit_operation
        """
        nums.sort()
        ceil = len(bin(max(nums))) - 1
        ans = mask = 0
        for i in range(ceil, -1, -1):
            new_ans = ans | (1 << i)
            pre = dict()
            mask |= (1 << i)
            for num in nums:
                cur = num & mask
                if cur ^ new_ans in pre and pre[cur ^ new_ans] * 2 >= num:
                    ans = new_ans
                    break
                pre[cur] = num
        return ans

    @staticmethod
    def cf_923c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/923/C
        tag: binary_trie|greedy
        """
        n = ac.read_int()
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        trie = BinaryTrieXor(1 << 30, n)
        for num in b:
            trie.add(num, 1)
        ans = []
        for num in a:
            ans.append(trie.get_minimum_xor(num))
            trie.remove(num ^ ans[-1])
        ac.lst(ans)
        return

    @staticmethod
    def cf_1055f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1055/problem/F
        tag: binary_trie|get_cnt_smaller_xor
        """
        n, k = ac.read_list_ints()
        dis = [0] * n
        dct = [[] for _ in range(n)]
        for i in range(n - 1):
            p, w = ac.read_list_ints()
            dct[p - 1].append((i + 1, w))
        stack = [0]
        while stack:
            x = stack.pop()
            for y, w in dct[x]:
                dis[y] = dis[x] ^ w
                stack.append(y)

        trie = BinaryTrieXor(max(dis) + 1, n)
        for num in dis:
            trie.add(num, 1)
        ans = (1 << 62) - 1
        for i in range(61, -1, -1):
            cnt = sum(trie.get_cnt_smaller_xor(x, ans ^ (1 << i)) for x in dis)
            if cnt >= k:
                ans ^= 1 << i
        ac.st(ans)
        return

    @staticmethod
    def cf_817e(ac=FastIO()):
        """
        url: https://codeforces.com/contest/817/problem/E
        tag: binary_trie|get_cnt_smaller_xor
        """
        q = ac.read_int()
        binary_trie = BinaryTrieXor(10 ** 8, q)
        for _ in range(q):
            lst = ac.read_list_ints()
            if lst[0] == 1:
                binary_trie.add(lst[1])
            elif lst[0] == 2:
                binary_trie.remove(lst[1])
            else:
                ac.st(binary_trie.get_cnt_smaller_xor(lst[1], lst[2] - 1))
        return
