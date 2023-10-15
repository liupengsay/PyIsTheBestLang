import heapq
import math
import random
import unittest
from src.fast_io import FastIO
from typing import List
from math import inf
from collections import Counter, defaultdict


class TrieZeroOneXorRange:
    def __init__(self, n):
        # 使用字典数据结构实现
        self.dct = dict()
        # 确定序列长度
        self.n = n
        return

    def update(self, num, cnt):
        cur = self.dct
        for i in range(self.n, -1, -1):
            # 更新当前哈希下的数字个数
            cur["cnt"] = cur.get("cnt", 0) + cnt
            # 哈希的键
            w = 1 if num & (1 << i) else 0
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        cur["cnt"] = cur.get("cnt", 0) + cnt
        return

    def query(self, num, ceil):

        def dfs(xor, cur, i):
            # 前缀异或和，当前哈希树，以及剩余序列所处的位数
            nonlocal res
            # 超出范围剪枝
            if xor > ceil:
                return
            # 搜索完毕剪枝
            if i == -1:
                res += cur["cnt"]
                return
            # 最大值也不超出范围剪枝
            if xor + (1 << (i + 2) - 1) <= ceil:
                res += cur["cnt"]
                return
            # 当前哈希的键
            w = 1 if num & (1 << i) else 0
            if 1 - w in cur:
                dfs(xor | (1 << i), cur[1 - w], i - 1)
            if w in cur:
                dfs(xor, cur[w], i - 1)
            return

        # 使用深搜查询异或值不超出范围的数对个数
        res = 0
        dfs(0, self.dct, self.n)
        return res


class TrieZeroOneXorMax:
    # 模板：使用01Trie维护与查询数组最大异或值
    def __init__(self, n):
        # 使用字典数据结构实现
        self.dct = dict()
        # 确定序列长度
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
        # 计算与num异或可以得到的最大值
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
    # 模板：更新单词的键值对与查询对应字符串前缀的值的和
    def __init__(self):
        self.dct = dict()
        return

    def update(self, word, val):
        # 更新单词与前缀计数
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        cur["val"] = val
        return

    def query(self, word):
        # 查询前缀单词个数
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
        # 更新单词与前缀计数
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
            cur["cnt"] = cur.get("cnt", 0) + 1
        return

    def query(self, word):
        # 查询前缀单词个数
        cur = self.dct
        for w in word:
            if w not in cur:
                return 0
            cur = cur[w]
        return cur["cnt"]


class TrieBit:
    def __init__(self, n=32):
        # 长度为n的二进制序列字典树
        self.dct = dict()
        self.n = n
        return

    # 加入新的值到字典树中
    def update(self, num):
        cur = self.dct
        for i in range(self.n, -1, -1):
            w = 1 if num & (1 << i) else 0
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
            cur["cnt"] = cur.get("cnt", 0) + 1
        return

    # 查询字典树最大异或值
    def query(self, num):
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

    # 从二进制序列中删除
    def delete(self, num):
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
    # 模板：更新单词集合，并计算给定字符串作为前缀的单词个数
    def __init__(self):
        self.dct = dict()
        return

    def update(self, word):
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
            cur['cnt'] = cur.get("cnt", 0) + 1
        return

    def query(self, word):
        cur = self.dct
        res = 0
        for w in word:
            if w not in cur:
                return False
            cur = cur[w]
            res += cur['cnt']
        return res


class TrieZeroOneXorMaxKth:
    # 模板：使用01Trie维护与查询数组的第 k 大异或值
    def __init__(self, n):
        # 使用字典数据结构实现
        self.dct = dict()
        # 确定序列长度
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
        # 计算与 num 异或可以得到的第 k 大值
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

