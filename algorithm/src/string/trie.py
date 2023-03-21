import math
import unittest
from algorithm.src.fast_io import FastIO
from typing import List
from collections import Counter

"""
算法：Trie字典树，也叫前缀树
功能：处理字符串以及结合位运算相关，01Trie通用用于查询位运算极值
题目：

===================================力扣===================================
2416. 字符串的前缀分数和（https://leetcode.cn/problems/sum-of-prefix-scores-of-strings/）单词组前缀计数
1803. 统计异或值在范围内的数对有多少（https://leetcode.cn/problems/count-pairs-with-xor-in-a-range/）经典01Trie，查询异或值在一定范围的数组对
677. 键值映射（https://leetcode.cn/problems/map-sum-pairs/）更新与查询给定字符串作为单词键前缀的对应值的和
2479. 两个不重叠子树的最大异或值（https://leetcode.cn/problems/maximum-xor-of-two-non-overlapping-subtrees/）01Trie计算最大异或值
面试题 17.17. 多次搜索（https://leetcode.cn/problems/multi-search-lcci/）AC自动机计数，也可直接使用字典树逆向思维，字典树存关键字，再搜索文本，和单词矩阵一样的套路
===================================洛谷===================================
P8306 字典树（https://www.luogu.com.cn/problem/P8306）
P4551 最长异或路径（https://www.luogu.com.cn/problem/P4551）关键是利用异或的性质，将任意根节点作为中转站
P3864 [USACO1.2]命名那个数字 Name That Number（https://www.luogu.com.cn/problem/P3864）使用哈希枚举或者进行字典树存储
P5755 [NOI2000] 单词查找树（https://www.luogu.com.cn/problem/P5755）字典树节点计数

================================CodeForces================================
Fixed Prefix Permutations（https://codeforces.com/problemset/problem/1792/D）变形后使用字典树进行计数查询
D. Vasiliy's Multiset（https://codeforces.com/problemset/problem/706/D）经典01Trie，增加与删除数字，最大异或值查询


参考：OI WiKi（）
"""


class Node:
    def __init__(self):
        self.data = 0
        self.left = None  # bit为0
        self.right = None  # bit为1
        self.count = 0


class TrieZeroOneXorNode:
    def __init__(self):
        # 使用自定义节点实现
        self.root = Node()
        self.cur = None
        self.n = 31

    def add(self, val):
        self.cur = self.root
        for i in range(self.n, -1, -1):
            v = val & (1 << i)
            if v:
                # 1 走右边
                if not self.cur.right:
                    self.cur.right = Node()
                self.cur = self.cur.right
                self.cur.count += 1
            else:
                # 0 走左边
                if not self.cur.left:
                    self.cur.left = Node()
                self.cur = self.cur.left
                self.cur.count += 1
        self.cur.data = val
        return

    def delete(self, val):
        self.cur = self.root
        for i in range(self.n, -1, -1):
            v = val & (1 << i)
            if v:
                # 1 走右边
                if self.cur.right.count == 1:
                    self.cur.right = None
                    break
                self.cur = self.cur.right
                self.cur.count -= 1
            else:
                # 0 走左边
                if self.cur.left.count == 1:
                    self.cur.left = None
                    break
                self.cur = self.cur.left
                self.cur.count -= 1
        return

    def query(self, val):
        self.cur = self.root
        for i in range(self.n, -1, -1):
            v = val & (1 << i)
            if v:
                # 1 优先走相反方向的左边
                if self.cur.left and self.cur.left.count > 0:
                    self.cur = self.cur.left
                elif self.cur.right and self.cur.right.count > 0:
                    self.cur = self.cur.right
            else:
                # 0 优先走相反方向的右边
                if self.cur.right and self.cur.right.count > 0:
                    self.cur = self.cur.right
                elif self.cur.left and self.cur.left.count > 0:
                    self.cur = self.cur.left
        return val ^ self.cur.data


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
        self.inf = float("inf")
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
    def __init__(self):
        self.dct = dict()
        self.n = 32
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
        cur = self.dct
        for i in range(self.n, -1, -1):
            w = num & (1 << i)
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

    
class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_1717(big: str, smalls: List[str]) -> List[List[int]]:
        # 模板：AC自动机类似题目，查询关键词在文本中的出现索引
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
        # 模板：更新与查询给定字符串作为单词键前缀的对应值的和
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
    def lc_1803(nums: List[int], low: int, high: int) -> int:
        # 模板：使用01字典树查询异或值在有一定范围内的数对个数
        count = Counter(nums)
        # 确定二进制序列的长度
        big = max(nums)
        n = 0
        while (1 << (n + 1)) - 1 < big:
            n += 1
        trie = TrieZeroOneXorRange(n)
        trie.update(0, 0)
        # 滚动更新字典树同时查询符合条件的数对个数
        ans = 0
        for num in count:
            ans += count[num]*(trie.query(num, high) - trie.query(num, low - 1))
            trie.update(num, count[num])
        return ans

    @staticmethod
    def cf_706d(ac=FastIO()):
        # 模板：使用01字典树增加与删除数字后查询最大异或值
        q = ac.read_int()
        trie = TrieZeroOneXorNode()
        trie.add(0)
        for _ in range(q):
            op, x = ac.read_list_strs()
            if op == "+":
                trie.add(int(x))
            elif op == "-":
                trie.delete(int(x))
            else:
                ac.st(trie.query(int(x)))
        return

    @staticmethod
    def lc_2479(n: int, edges: List[List[int]], values: List[int]) -> int:
        # 模板：借助深搜的顺序进行01字典树查询最大异或数对值
        dct = [[] for _ in range(n)]
        for i, j in edges:
            dct[i].append(j)
            dct[j].append(i)

        # 预处理子树取值之和
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
        # 根据题意使用深搜序和01字典树动态维护查询
        trie = TrieZeroOneXorMax(int(math.log2(sum(values)) + 2))
        ans = 0
        dfs2(0, -1)
        return ans
    
    @staticmethod
    def lc_2416(words: List[str]) -> List[int]:
        trie = TriePrefixCount()
        for word in words:
            trie.update(word)
        return [trie.query(word) for word in words]



class TestGeneral(unittest.TestCase):

    def test_trie_count(self):
        tc = TrieCount()
        words = ["happy", "hello", "leetcode", "let"]
        for word in words:
            tc.update(word)
        assert tc.query("h") == 2
        assert tc.query("le") == 2
        assert tc.query("lt") == 0
        return


if __name__ == '__main__':
    unittest.main()
