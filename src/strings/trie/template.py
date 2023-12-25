import random

from src.utils.fast_io import inf


class TrieZeroOneXorRange:
    def __init__(self, n):
        self.dct = dict()
        self.n = n
        return

    def update(self, num, cnt):
        cur = self.dct
        for i in range(self.n, -1, -1):
            # update cnt of subtree number
            cur["cnt"] = cur.get("cnt", 0) + cnt
            w = 1 if num & (1 << i) else 0
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        cur["cnt"] = cur.get("cnt", 0) + cnt
        return

    def query(self, num, ceil):

        def dfs(xor, cur, i):
            # params are prefix xor and cur trie and i-th bit
            nonlocal res
            # prune by out of range
            if xor > ceil:
                return
            # done
            if i == -1:
                res += cur["cnt"]
                return
            # prune by ceil
            if xor + (1 << (i + 2) - 1) <= ceil:
                res += cur["cnt"]
                return
            w = 1 if num & (1 << i) else 0
            if 1 - w in cur:
                dfs(xor | (1 << i), cur[1 - w], i - 1)
            if w in cur:
                dfs(xor, cur[w], i - 1)
            return

        # check the cnt of number with limit range of xor
        res = 0
        dfs(0, self.dct, self.n)
        return res


class TrieZeroOneXorMax:
    """template of add num and query the maximum xor with assign num"""

    def __init__(self, n):
        self.dct = dict()
        self.n = n
        self.inf = inf
        return

    def add(self, num):
        cur = self.dct
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        return

    def query_xor_max(self, num):
        cur = self.dct
        ans = 0
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if 1 - w in cur:
                cur = cur[1 - w]
                ans |= (1 << i)
            elif w in cur:
                cur = cur[w]
            else:
                return 0
        return ans


class TriePrefixKeyValue:
    """template of word prefix cnt and query"""

    def __init__(self):
        self.dct = dict()
        return

    def update(self, word, val):
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        cur["val"] = val
        return

    def query(self, word):
        cur = self.dct
        for w in word:
            if w not in cur:
                return 0
            cur = cur[w]

        def dfs(dct):
            nonlocal res
            if "val" in dct:
                res += dct["val"]
            for s in dct:
                if s != "val":
                    dfs(dct[s])
            return

        res = 0
        dfs(cur)
        return res


class TrieCount:
    def __init__(self):
        self.dct = dict()
        return

    def update(self, word):
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
            cur["cnt"] = cur.get("cnt", 0) + 1
        return

    def query(self, word):
        cur = self.dct
        for w in word:
            if w not in cur:
                return 0
            cur = cur[w]
        return cur["cnt"]


class TrieBit:
    def __init__(self, n=32):
        """template of add and remove num for maximum xor query"""
        self.dct = dict()
        self.n = n
        return

    def update(self, num):
        cur = self.dct
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
            cur["cnt"] = cur.get("cnt", 0) + 1
        return

    def query(self, num):
        """query maximum xor value"""
        cur = self.dct
        ans = 0
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if 1 - w in cur:
                cur = cur[1 - w]
                ans += 1 << i
            else:
                cur = cur[w]
        return ans

    def delete(self, num):
        """remove one num"""
        cur = self.dct
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if cur[w].get("cnt", 0) == 1:
                del cur[w]
                break
            cur = cur[w]
            cur["cnt"] -= 1
        return


class TrieKeyWordSearchInText:
    def __init__(self):
        self.dct = dict()
        return

    def add_key_word(self, word, i):
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        cur["isEnd"] = i

    def search_text(self, text):
        cur = self.dct
        res = []
        for w in text:
            if w in cur:
                cur = cur[w]
                if "isEnd" in cur:
                    res.append(cur["isEnd"])
            else:
                break
        return res


class TriePrefixCount:
    """use array and dict to produce trie"""

    def __init__(self):
        self.seed = random.randint(0, 10 ** 9 + 7)
        self.ind = dict()
        self.cnt = [0]
        return

    def update(self, word):
        i = 0
        for w in word:
            j = i * 26 + ord(w) - ord("a")
            if j ^ self.seed not in self.ind:
                self.ind[j ^ self.seed] = len(self.cnt)
                self.cnt.append(0)
            i = self.ind[j ^ self.seed]
            self.cnt[i] += 1
        return

    def query(self, word):
        res = i = 0
        for w in word:
            j = i * 26 + ord(w) - ord("a")
            if j ^ self.seed not in self.ind:
                break
            i = self.ind[j ^ self.seed]
            res += self.cnt[i]
        return res


class TrieZeroOneXorMaxKth:
    def __init__(self, n):
        """template of query the k-th xor value of assign num"""
        self.dct = dict()
        self.n = n
        self.inf = inf
        return

    def add(self, num):
        cur = self.dct
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
            cur["cnt"] = cur.get("cnt", 0) + 1
        return

    def query_xor_kth_max(self, num, k):
        cur = self.dct
        ans = 0
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if 1 - w in cur and cur[1 - w]["cnt"] >= k:
                cur = cur[1 - w]
                ans |= (1 << i)
            else:
                if 1 - w in cur:
                    k -= cur[1 - w].get("cnt", 0)
                cur = cur[w]
        return ans


class BinaryTrie:
    def __init__(self, max_bit: int = 30):
        self.inf = 1 << 63
        self.to = [[-1], [-1]]
        self.cnt = [0]
        self.max_bit = max_bit

    def add(self, num: int) -> None:
        cur = 0
        self.cnt[cur] += 1
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            if self.to[bit][cur] == -1:
                self.to[bit][cur] = len(self.cnt)
                self.to[0].append(-1)
                self.to[1].append(-1)
                self.cnt.append(0)
            cur = self.to[bit][cur]
            self.cnt[cur] += 1
        return

    def remove(self, num: int) -> bool:
        if self.cnt[0] == 0:
            return False
        cur = 0
        rm = [0]
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            cur = self.to[bit][cur]
            if cur == -1 or self.cnt[cur] == 0:
                return False
            rm.append(cur)
        for cur in rm:
            self.cnt[cur] -= 1
        return True

    def count(self, num: int):
        cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (num >> k) & 1
            cur = self.to[bit][cur]
            if cur == -1 or self.cnt[cur] == 0:
                return 0
        return self.cnt[cur]

    # Get max result for constant x ^ element in array
    def max_xor(self, x: int) -> int:
        if self.cnt[0] == 0:
            return -self.inf
        res = cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            nxt = self.to[bit ^ 1][cur]
            if nxt == -1 or self.cnt[nxt] == 0:
                cur = self.to[bit][cur]
            else:
                res |= 1 << k
                cur = nxt

        return res

    # Get min result for constant x ^ element in array
    def min_xor(self, x: int) -> int:
        if self.cnt[0] == 0:
            return self.inf
        res = cur = 0
        for k in range(self.max_bit, -1, -1):
            bit = (x >> k) & 1
            nxt = self.to[bit][cur]
            if nxt == -1 or self.cnt[nxt] == 0:
                res |= 1 << k
                cur = self.to[bit ^ 1][cur]
            else:
                cur = nxt
        return res
