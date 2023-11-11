from typing import Optional

from src.data_structure.list_node.template import ListNode, ListNodeOperation

"""
算法：链表、两个链表表示的整数相加、链表表示的整数与整数相乘
功能：
参考：
题目：

===================================力扣===================================
6914. 翻倍以链表形式表示的数字（https://leetcode.cn/contest/weekly-contest-358/problems/double-a-number-represented-as-a-linked-list/）链表形式的数字与整数相乘

"""


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_6914_1(head: Optional[ListNode]) -> Optional[ListNode]:
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

    @staticmethod
    def lc_6914_2(head: Optional[ListNode]) -> Optional[ListNode]:
        lno = ListNodeOperation()
        num = lno.node_to_num(head) * 2
        return lno.num_to_node(num)
