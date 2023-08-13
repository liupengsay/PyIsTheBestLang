import unittest
from typing import List, Optional

"""
算法：链表、两个链表表示的整数相加、链表表示的整数与整数相乘
功能：
参考：
题目：

===================================力扣===================================
6914. 翻倍以链表形式表示的数字（https://leetcode.cn/contest/weekly-contest-358/problems/double-a-number-represented-as-a-linked-list/）链表形式的数字与整数相乘

"""


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


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_6914(head: Optional[ListNode]) -> Optional[ListNode]:
        # 模板：链表与整数相乘
        lno = ListNodeOperation()
        lst = lno.node_to_lst(head)[::-1]

        nums = []
        x = 0
        for num in lst:
            x += num * 2
            nums.append(x % 10)
            x = 1 if x >= 10 else 0
        if x:
            nums.append(x)

        nums.reverse()
        return lno.lst_to_node(nums)


class TestGeneral(unittest.TestCase):

    def test_xxxx(self):
        pass
        return


if __name__ == '__main__':
    unittest.main()
