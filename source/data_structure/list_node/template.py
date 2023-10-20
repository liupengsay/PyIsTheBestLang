"""
算法：链表、两个链表表示的整数相加、链表表示的整数与整数相乘
功能：
参考：
题目：

===================================力扣===================================
6914. 翻倍以链表形式表示的数字（https://leetcode.cn/contest/weekly-contest-358/problems/double-a-number-represented-as-a-linked-list/）链表形式的数字与整数相乘

"""
from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class ListNodeOperation:
    def __init__(self):
        return

    @staticmethod
    def node_to_lst(node: ListNode) -> List[int]:
        lst = []
        while node:
            lst.append(node.val)
            node = node.next
        return lst

    @staticmethod
    def lst_to_node(lst: List[int]) -> ListNode:
        node = ListNode(-1)
        pre = node
        for num in lst:
            pre.next = ListNode(num)
            pre = pre.next
        return node.next

    @staticmethod
    def node_to_num(node: ListNode) -> int:
        num = 0
        while node:
            num = num * 10 + node.val
            node = node.next
        return num

    @staticmethod
    def num_to_node(num: int) -> ListNode:
        node = ListNode(-1)
        pre = node
        for x in str(num):
            pre.next = ListNode(int(x))
            pre = pre.next
        return node.next
