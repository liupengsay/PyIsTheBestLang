"""
算法：AC自动机
功能：KMP加Trie的结合应用，用关键词建立字典树，再查询文本中关键词的出现次数
题目：

===================================力扣===================================
面试题 17.17. 多次搜索（https://leetcode.com/problems/multi-search-lcci/）AC自动机计数，也可直接使用字典树

===================================洛谷===================================
P3808 【模板】AC 自动机（简单版）（https://www.luogu.com.cn/problem/P3808）AC自动机计数
P3796 【模板】AC自动机（加强版）（https://www.luogu.com.cn/problem/P3796）AC自动机计数
P5357 【模板】AC自动机（二次加强版）（https://www.luogu.com.cn/problem/P5357）AC自动机计数

参考：OI WiKi（xx）
"""

from typing import List

from src.strings.ac_auto_machine.template import AhoCorasick


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1717(big: str, smalls: List[str]) -> List[List[int]]:
        # 模板：AC自动机匹配关键词在文本中出现的位置信息
        auto = AhoCorasick(smalls)
        dct = auto.search_in(big)
        return [dct.get(w, []) for w in smalls]
