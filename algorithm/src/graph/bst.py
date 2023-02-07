
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

"""
算法：BST二叉搜索树
功能：根据数字顺序建立二叉搜索树、实时维护
题目：

===================================洛谷===================================
P2171 Hz吐泡泡（https://www.luogu.com.cn/problem/P2171）依次输入数据生成二叉搜索树

参考：OI WiKi（xx）
"""

ans = 1  # 标记是第几层
ans1 = 1  # 最高层数


class BST(object):
    def __init__(self, data, left=None, right=None):  # BST的三个，值，左子树，右子树
        # 这里遇到链条形状的树会有性能问题
        super(BST, self).__init__()
        self.data = data
        self.left = left
        self.right = right

    def insert(self, val):  # 插入函数
        global ans, ans1
        if val < self.data:  # 如果小于就放到左子树
            if self.left:  # 如果有左子树
                ans += 1  # 层数+1
                self.left.insert(val)  # 左子树调用递归
            else:  # 没有左子树
                ans += 1  # 层数+1
                self.left = BST(val)  # 把值放在这个点的左子树上
                if ans > ans1:  # 如果层数比之前最高层数高
                    ans1 = ans  # 替换
                ans = 1  # 重新开始
        else:  # 比节点的值大
            if self.right:  # 有右子树
                ans += 1  # 层数+1
                self.right.insert(val)  # 右子树调用递归
            else:  # 没有右子树
                ans += 1  # 层数+1
                self.right = BST(val)  # 将值放在右子树上
                if ans > ans1:  # 如果层数大于最高层数
                    ans1 = ans  # 覆盖
                ans = 1  # 重新开始
        return

    def post_order(self):  # 后序遍历：左，右，根
        if self.left:  # 有左子树
            self.left.post_order()  # 先遍历它的左子树
        if self.right:  # 有右子树
            self.right.post_order()  # 遍历右子树
        print(self.data)  # 都没有或已经遍历完就输出值
        return


class Node:
    # Constructor assigns the given key, with left and right
    # children assigned with None.
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BinarySearchTree:
    # Constructor just assigns an empty root.
    def __init__(self):
        self.root = None

    # Search for a node containing a matching key. Returns the
    # Node object that has the matching key if found, None if
    # not found.
    def search(self, desired_key):
        current_node = self.root
        while current_node is not None:
            # Return the node if the key matches.
            if current_node.key == desired_key:
                return current_node

            # Navigate to the left if the search key is
            # less than the node's key.
            elif desired_key < current_node.key:
                current_node = current_node.left

            # Navigate to the right if the search key is
            # greater than the node's key.
            else:
                current_node = current_node.right

        # The key was not found in the tree.
        return None

    # Inserts the new node into the tree.
    def insert(self, node):

        # Check if the tree is empty
        if self.root is None:
            self.root = node
        else:
            current_node = self.root
            while current_node is not None:
                if node.key < current_node.key:
                    # If there is no left child, add the new
                    # node here; otherwise repeat from the
                    # left child.
                    if current_node.left is None:
                        current_node.left = node
                        current_node = None
                    else:
                        current_node = current_node.left
                else:
                    # If there is no right child, add the new
                    # node here; otherwise repeat from the
                    # right child.
                    if current_node.right is None:
                        current_node.right = node
                        current_node = None
                    else:
                        current_node = current_node.right

                        # Removes the node with the matching key from the tree.

    def remove(self, key):
        parent = None
        current_node = self.root

        # Search for the node.
        while current_node is not None:

            # Check if current_node has a matching key.
            if current_node.key == key:
                if current_node.left is None and current_node.right is None:  # Case 1
                    if parent is None:  # Node is root
                        self.root = None
                    elif parent.left is current_node:
                        parent.left = None
                    else:
                        parent.right = None
                    return  # Node found and removed
                elif current_node.left is not None and current_node.right is None:  # Case 2
                    if parent is None:  # Node is root
                        self.root = current_node.left
                    elif parent.left is current_node:
                        parent.left = current_node.left
                    else:
                        parent.right = current_node.left
                    return  # Node found and removed
                elif current_node.left is None and current_node.right is not None:  # Case 2
                    if parent is None:  # Node is root
                        self.root = current_node.right
                    elif parent.left is current_node:
                        parent.left = current_node.right
                    else:
                        parent.right = current_node.right
                    return  # Node found and removed
                else:  # Case 3
                    # Find successor (leftmost child of right subtree)
                    successor = current_node.right
                    while successor.left is not None:
                        successor = successor.left
                    current_node.key = successor.key  # Copy successor to current node
                    parent = current_node
                    current_node = current_node.right  # Remove successor from right subtree
                    key = parent.key  # Loop continues with new key
            elif current_node.key < key:  # Search right
                parent = current_node
                current_node = current_node.right
            else:  # Search left
                parent = current_node
                current_node = current_node.left

        return  # Node not found


class TestGeneral(unittest.TestCase):

    def test_xxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
