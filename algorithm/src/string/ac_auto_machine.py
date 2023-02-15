"""

"""
"""
算法：AC自动机
功能：KMP加Trie的结合应用，用关键词建立字典树，再查询文本中关键词的出现次数
题目：

===================================洛谷===================================
P3808 【模板】AC 自动机（简单版）（https://www.luogu.com.cn/problem/P3808）AC自动机计数
P3796 【模板】AC自动机（加强版）（https://www.luogu.com.cn/problem/P3796）AC自动机计数
P5357 【模板】AC自动机（二次加强版）（https://www.luogu.com.cn/problem/P5357）AC自动机计数

参考：OI WiKi（xx）
"""

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy

from queue import Queue
from typing import List, Dict, Iterable


class AhoCorasick(object):

    class Node(object):

        def __init__(self, name: str):
            self.name = name  # 节点代表的字符
            self.children = {}  # 节点的孩子，键为字符，值为节点对象
            self.fail = None  # fail指针，root的指针为None
            self.exist = []  # 如果节点为单词结尾，存放单词的长度

    def __init__(self, keywords: Iterable[str] = None):
        """AC自动机"""
        self.root = self.Node("root")
        self.finalized = False
        if keywords is not None:
            for keyword in set(keywords):
                self.add(keyword)

    def add(self, keyword: str):
        if self.finalized:
            raise RuntimeError('The tree has been finalized!')
        node = self.root
        for char in keyword:
            if char not in node.children:
                node.children[char] = self.Node(char)
            node = node.children[char]
        node.exist.append(len(keyword))

    def contains(self, keyword: str) -> bool:
        node = self.root
        for char in keyword:
            if char not in node.children:
                return False
            node = node.children[char]
        return bool(node.exist)

    def finalize(self):
        """构建fail指针"""
        queue = Queue()
        queue.put(self.root)
        # 对树进行层次遍历
        while not queue.empty():
            node = queue.get()
            for char in node.children:
                child = node.children[char]
                f_node = node.fail
                # 关键点！需要沿着fail指针向上追溯直至根节点
                while f_node is not None:
                    if char in f_node.children:
                        # 如果该指针指向的节点的孩子中有该字符，则字符节点的fail指针需指向它
                        f_child = f_node.children[char]
                        child.fail = f_child
                        # 同时将长度合并过来，以便最后输出
                        if f_child.exist:
                            child.exist.extend(f_child.exist)
                        break
                    f_node = f_node.fail
                # 如果到根节点也没找到，则将fail指针指向根节点
                if f_node is None:
                    child.fail = self.root
                queue.put(child)
        self.finalized = True

    def search_in(self, text: str) -> Dict[str, List[int]]:
        """在一段文本中查找关键字及其开始位置（可能重复多个）"""
        result = dict()
        if not self.finalized:
            self.finalize()
        node = self.root
        for i, char in enumerate(text):
            matched = True
            # 如果当前节点的孩子中找不到该字符
            while char not in node.children:
                # fail指针为None，说明走到了根节点，找不到匹配的
                if node.fail is None:
                    matched = False
                    break
                # 将fail指针指向的节点作为当前节点
                node = node.fail
            if matched:
                # 找到匹配，将匹配到的孩子节点作为当前节点
                node = node.children[char]
                if node.exist:
                    # 如果该节点存在多个长度，则输出多个关键词
                    for length in node.exist:
                        start = i - length + 1
                        word = text[start: start + length]
                        if word not in result:
                            result[word] = []
                        result[word].append(start)
        return result



from collections import defaultdict

class TrieNode():
    def __init__(self):
        self.child = {}
        self.failto = None
        self.is_word = False
        '''
        下面节点值可以根据具体场景进行赋值
        '''
        self.str_ = ''
        self.num = 0

class AhoCorasickAutomation:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def buildTrieTree(self, wordlst):
        for word in wordlst:
            cur = self.root
            for i, c in enumerate(word):
                if c not in cur.child:
                    cur.child[c] = TrieNode()
                ps = cur.str_
                cur = cur.child[c]
                cur.str_ = ps + c
            cur.is_word = True
            cur.num += 1

    def build_AC_from_Trie(self):
        queue = []
        for child in self.root.child:
            self.root.child[child].failto = self.root
            queue.append(self.root.child[child])

        while len(queue) > 0:
            cur = queue.pop(0)
            for child in cur.child.keys():
                failto = cur.failto
                while True:
                    if failto == None:
                        cur.child[child].failto = self.root
                        break
                    if child in failto.child:
                        cur.child[child].failto = failto.child[child]
                        break
                    else:
                        failto = failto.failto
                queue.append(cur.child[child])

    def ac_search(self, str_):
        cur = self.root
        result = defaultdict(int)
        # result = {}
        i = 0
        n = len(str_)
        while i < n:
            c = str_[i]
            if c in cur.child:
                cur = cur.child[c]
                if cur.is_word:
                    temp = cur.str_
                    result[temp] += 1
                    # result.setdefault(temp, [])
                    # result[temp].append([i - len(temp) + 1, i, cur.num])

                '''
                处理所有其他长度公共字串
                '''
                fl = cur.failto
                while fl:
                    if fl.is_word:
                        temp = fl.str_
                        result[temp] += 1
                        # result.setdefault(temp, [])
                        # result[temp].append([i - len(temp) + 1, i, cur.failto.num])
                    fl = fl.failto
                i += 1

            else:
                cur = cur.failto
                if cur == None:
                    cur = self.root
                    i += 1
        return result


class TestGeneral(unittest.TestCase):

    def test_AhoCorasick(self):

        keywords = ["i","is", "ssippi"]
        auto = AhoCorasick(keywords)
        text = "misissippi"
        assert auto.search_in(text) == {"i": [1, 3, 6, 9], "is": [1, 3], "ssippi": [4]}

        acTree = AhoCorasickAutomation()
        acTree.buildTrieTree(["i", "is", "ssippi"])
        acTree.build_AC_from_Trie()
        assert acTree.ac_search("misissippi") == {"i": 4, "is": 2, "ssippi": 1}
        return



if __name__ == '__main__':
    unittest.main()
