"""
Algorithm：automaton|sub_sequence_automaton|suffix_automaton|palindrome_automaton
Description：kmp|trie_like|word_count|text

====================================LeetCode====================================
1717（https://leetcode.cn/problems/multi-search-lcci/）ac_automaton|counter|trie_like
727（https://leetcode.cn/problems/minimum-window-subsequence）sub_sequence_automaton
792（https://leetcode.cn/problems/number-of-matching-subsequences）sub_sequence_automaton
2014（https://leetcode.cn/problems/longest-subsequence-repeated-k-times/）sub_sequence_automaton|brute_force
2350（https://leetcode.cn/problems/shortest-impossible-sequence-of-rolls/）brain_teaser|classical|sub_sequence_automaton
3292（https://leetcode.cn/problems/minimum-number-of-valid-strings-to-form-target-ii）ac_automaton|AhoCorasick
3213（https://leetcode.cn/problems/construct-string-with-minimum-cost/）ac_automaton|AhoCorasick

=====================================LuoGu======================================
P3808（https://www.luogu.com.cn/problem/P3808）ac_automan|classical
P3796（https://www.luogu.com.cn/problem/P3796）ac_automan|classical
P5357（https://www.luogu.com.cn/problem/P5357）ac_automan|classical
P5826（https://www.luogu.com.cn/problem/P5826）sub_sequence_automaton|binary_search
P9572（https://www.luogu.com.cn/problem/P9572）sub_sequence_automaton|binary_search

=====================================AtCoder======================================
ABC362G（https://atcoder.jp/contests/abc362/tasks/abc362_g）ac_automan|classical

===================================CodeForces===================================
91A（https://codeforces.com/contest/91/problem/A）sub_sequence_automaton
1845C（https://codeforces.com/contest/1845/problem/C）sub_sequence_automaton

"""
import bisect
from collections import Counter
from itertools import permutations

from typing import List

from src.strings.automaton.template import AcAutomaton, AhoCorasick
from src.utils.fast_io import FastIO


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1717(big: str, smalls: List[str]) -> List[List[int]]:
        """
        url: https://leetcode.cn/problems/construct-string-with-minimum-cost/
        tag: ac_automaton|AhoCorasick
        """
        ac = AhoCorasick()
        for i, word in enumerate(smalls):
            ac.insert(word, i)
        ac.set_fail()
        res = ac.search(big)
        ans = [[] for _ in range(len(smalls))]
        for i in range(len(big)):
            for j in res[i]:
                ans[j].append(i)
        return ans

    @staticmethod
    def lg_p5826(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5826
        tag: sub_sequence_automaton|binary_search
        """
        _, n, q, m = ac.read_list_ints()
        ind = [[] for _ in range(m + 1)]
        lst = ac.read_list_ints()
        for i, num in enumerate(lst):
            ind[num].append(i)
        for _ in range(q):
            lst = ac.read_list_ints()[1:]
            i = 0
            for w in lst:
                j = bisect.bisect_left(ind[w], i)
                if j >= len(ind[w]):
                    ac.no()
                    break
                i = ind[w][j] + 1
            else:
                ac.yes()
        return

    @staticmethod
    def lc_727(s1: str, s2: str) -> str:
        """
        url: https://leetcode.cn/problems/minimum-window-subsequence
        tag: sub_sequence_automaton
        """
        lst1 = [ord(w) - ord("a") for w in s1]
        lst2 = [ord(w) - ord("a") for w in s2]
        m, n = len(s1), len(s2)

        nxt = [-1] * 26 * m
        post = dict()
        for i in range(m - 1, -1, -1):
            post[lst1[i]] = i
            for j in post:
                nxt[i * 26 + j] = post[j]

        ans = [0, m]
        for i in range(m):
            j = 0
            k = i
            while j < n and k < m:
                k = nxt[k * 26 + lst2[j]]
                if k == -1:
                    break
                k += 1
                j += 1
            if j == n and k - i < ans[1] - ans[0] + 1:
                ans = [i, k - 1]
        if ans == [0, m]:
            return ""
        return s1[ans[0]:ans[1] + 1]

    @staticmethod
    def lc_792(s: str, words: List[str]) -> int:
        """
        url: https://leetcode.cn/problems/number-of-matching-subsequences/
        tag: sub_sequence_automaton
        """
        lst = [ord(w) - ord("a") for w in s]
        m = len(s)

        nxt = [-1] * 26 * m
        post = dict()
        for i in range(m - 1, -1, -1):
            post[lst[i]] = i
            for j in post:
                nxt[i * 26 + j] = post[j]

        ans = 0
        for word in words:
            lst = [ord(w) - ord("a") for w in word]
            n = len(word)

            j = 0
            k = 0
            while j < n and k < m:
                k = nxt[k * 26 + lst[j]]
                if k == -1:
                    break
                k += 1
                j += 1
            if j == n:
                ans += 1
        return ans

    @staticmethod
    def lc_2014(s: str, k: int) -> str:
        """
        url: https://leetcode.cn/problems/longest-subsequence-repeated-k-times/
        tag: sub_sequence_automaton|brute_force
        """
        lst = [ord(w) - ord("a") for w in s]
        n = len(s)
        nxt = [-1] * 26 * n
        post = dict()
        for i in range(n - 1, -1, -1):
            post[lst[i]] = i
            for j in post:
                nxt[i * 26 + j] = post[j]

        cnt = Counter(lst)
        hot = []
        for w in sorted(cnt, reverse=True):
            hot.extend([w] * (cnt[w] // k))

        pre = set()
        for m in range(min(len(hot), 7), 0, -1):
            for item in permutations(hot, m):
                if item in pre:
                    continue
                pre.add(item)
                i = j = 0
                while j < k * m and i < n:
                    i = nxt[i * 26 + item[j % m]]
                    if i == -1:
                        break
                    i += 1
                    j += 1
                if j == k * m:
                    word = "".join([chr(x + ord("a")) for x in item])
                    return word
        return ""

    @staticmethod
    def cf_91a(ac=FastIO()):
        """
        url: https://codeforces.com/contest/91/problem/A
        tag: sub_sequence_automaton
        """
        s = [ord(w) - ord("a") for w in ac.read_str()]
        t = [ord(w) - ord("a") for w in ac.read_str()]
        n, m = len(s), len(t)

        nxt = [-1] * 26 * n
        post = dict()
        for i in range(n - 1, -1, -1):
            post[s[i]] = i

        for i in range(n - 1, -1, -1):
            for j in post:
                nxt[i * 26 + j] = post[j]
            post[s[i]] = i

        if nxt[(n - 1) * 26 + t[0]] == -1:
            ac.st(-1)
            return

        i = nxt[(n - 1) * 26 + t[0]]
        ans = 1
        for j in t[1:]:
            k = nxt[i * 26 + j]
            if k == -1:
                ac.st(-1)
                return
            if k <= i:
                ans += 1
            i = k
        ac.st(ans)
        return

    @staticmethod
    def lg_p9572(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P9572
        tag: sub_sequence_automaton
        """
        n, m, c1, c2 = ac.read_list_ints()
        s = ac.read_list_ints()
        dct = dict()
        for i, w in enumerate(s):
            if w not in dct:
                dct[w] = []
            dct[w].append(i)
        t = [w for w in ac.read_list_ints() if w in dct]
        if not t:
            ac.lst([0, 0])
            return
        ans = 1
        i = dct[t[0]][0]
        for x in t[1:]:
            if dct[x][-1] > i:
                i = dct[x][bisect.bisect_right(dct[x], i)]
            else:
                ans += 1
                i = dct[x][0]
        ac.lst([c1 * len(t), c2 * ans])
        return

    @staticmethod
    def cf_1845c(ac=FastIO()):

        """
        url: https://codeforces.com/contest/1845/problem/C
        tag: sub_sequence_automaton
        """

        def check():
            i = 0
            ll, rr = int(s1[i]), int(s2[i])
            pre = set()
            for w in s:
                if ll <= int(w) <= rr:
                    pre.add(int(w))
                if len(pre) == rr - ll + 1:
                    pre = set()
                    i += 1
                    if i == m:
                        ac.no()
                        return
                    ll, rr = int(s1[i]), int(s2[i])
            ac.yes()
            return

        for _ in range(ac.read_int()):
            s = ac.read_str()
            m = ac.read_int()
            s1 = ac.read_str()
            s2 = ac.read_str()
            check()
        return

    @staticmethod
    def lc_2350(rolls: List[int], k: int) -> int:
        """
        url: https://leetcode.cn/problems/shortest-impossible-sequence-of-rolls/
        tag: brain_teaser|classical|sub_sequence_automaton
        """
        ans = 1
        pre = set()
        for num in rolls:
            pre.add(num)
            if len(pre) == k:
                ans += 1
                pre = set()
        return ans

    @staticmethod
    def abc_362g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc362/tasks/abc362_g
        tag: ac_automan|classical
        """
        s = ac.read_str()
        q = ac.read_int()
        words = [ac.read_str() for _ in range(q)]
        am = AcAutomaton(words)
        res = am.search(s)
        for a in res:
            ac.st(a)
        return

    @staticmethod
    def lg_p5357(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5357
        tag: ac_automan|classical
        """
        q = ac.read_int()
        words = [ac.read_str() for _ in range(q)]
        s = ac.read_str()
        am = AcAutomaton(words)
        res = am.search(s)
        for a in res:
            ac.st(a)
        return

    @staticmethod
    def lg_p3808(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3808
        tag: ac_automan|classical
        """
        q = ac.read_int()
        words = [ac.read_str() for _ in range(q)]
        s = ac.read_str()
        am = AcAutomaton(words)
        res = am.search(s)
        ans = sum(x > 0 for x in res)
        ac.st(ans)
        return

    @staticmethod
    def lg_p3796(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3796
        tag: ac_automan|classical
        """
        while True:
            q = ac.read_int()
            if not q:
                break
            words = [ac.read_str() for _ in range(q)]
            s = ac.read_str()
            am = AcAutomaton(words)
            res = am.search(s)
            i = res.index(max(res))
            ac.st(res[i])
            for j in range(q):
                if res[j] == res[i]:
                    ac.st(words[j])
        return

    @staticmethod
    def lc_3213(target: str, words: List[str], costs: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/construct-string-with-minimum-cost/
        tag: ac_automaton|AhoCorasick
        """
        ac = AhoCorasick()
        for word, cost in zip(words, costs):
            ac.insert(word, cost)
        ac.set_fail()
        f = [0] + [math.inf] * len(target)
        x = ac.root
        for i, c in enumerate(target, 1):
            while x and c not in x.son:
                x = x.fail
            x = x.son[c] if x else ac.root
            cur = x
            while cur:
                if cur.len:
                    tmp = f[i - cur.len] + cur.val
                    if f[i] > tmp:
                        f[i] = tmp
                cur = cur.last
        return f[-1] if f[-1] < inf else -1