from collections import defaultdict

from queue import Queue
from typing import List, Dict, Iterable


class AhoCorasick(object):
    class Node(object):

        def __init__(self, name: str):
            self.name = name  # 节点代表的字符
            self.children = {}  # 节点的孩子，键为字符，值为节点对象
            self.fail = None  # failpointer，root的pointer为None
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
        """构建failpointer"""
        queue = Queue()
        queue.put(self.root)
        # 对树层次遍历
        while not queue.empty():
            node = queue.get()
            for char in node.children:
                child = node.children[char]
                f_node = node.fail
                # 关键点！需要沿着failpointer向上追溯直至根节点
                while f_node is not None:
                    if char in f_node.children:
                        # 如果该pointer指向的节点的孩子中有该字符，则字符节点的failpointer需指向它
                        f_child = f_node.children[char]
                        child.fail = f_child
                        # 同时将长度合并过来，以便最后输出
                        if f_child.exist:
                            child.exist.extend(f_child.exist)
                        break
                    f_node = f_node.fail
                # 如果到根节点也没找到，则将failpointer指向根节点
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
                # failpointer为None，说明走到了根节点，找不到匹配的
                if node.fail is None:
                    matched = False
                    break
                # 将failpointer指向的节点作为当前节点
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


class TrieNode:
    def __init__(self):
        self.child = {}
        self.fail_to = None
        self.is_word = False
        '''
        下面节点值可以根据具体场景赋值
        '''
        self.str_ = ''
        self.num = 0


class AhoCorasickAutomation:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def build_trie_tree(self, word_lst):
        for word in word_lst:
            cur = self.root
            for i, c in enumerate(word):
                if c not in cur.child:
                    cur.child[c] = TrieNode()
                ps = cur.str_
                cur = cur.child[c]
                cur.str_ = ps + c
            cur.is_word = True
            cur.num += 1

    def build_ac_from_trie(self):
        queue = []
        for child in self.root.child:
            self.root.child[child].fail_to = self.root
            queue.append(self.root.child[child])

        while len(queue) > 0:
            cur = queue.pop(0)
            for child in cur.child.keys():
                fail_to = cur.fail_to
                while True:
                    if not fail_to:
                        cur.child[child].fail_to = self.root
                        break
                    if child in fail_to.child:
                        cur.child[child].fail_to = fail_to.child[child]
                        break
                    else:
                        fail_to = fail_to.fail_to
                queue.append(cur.child[child])

    def ac_search(self, str_):
        cur = self.root
        result = defaultdict(int)
        dct = {}  # 输出具体索引
        i = 0
        n = len(str_)
        while i < n:
            c = str_[i]
            if c in cur.child:
                cur = cur.child[c]
                if cur.is_word:
                    temp = cur.str_
                    result[temp] += 1
                    dct.setdefault(temp, [])
                    dct[temp].append(i - len(temp) + 1)

                '''
                处理所有其他长度公共字串
                '''
                fl = cur.fail_to
                while fl:
                    if fl.is_word:
                        temp = fl.str_
                        result[temp] += 1
                        dct.setdefault(temp, [])
                        dct[temp].append(i - len(temp) + 1)
                    fl = fl.fail_to
                i += 1

            else:
                cur = cur.fail_to
                if not cur:
                    cur = self.root
                    i += 1
        return result, dct
