"""
Algorithm：ac_auto_machine
Description：kmp|trie|word_count|text

====================================LeetCode====================================
17（https://leetcode.com/problems/multi-search-lcci/）ac_auto_machine|counter|trie

=====================================LuoGu======================================
P3808（https://www.luogu.com.cn/problem/P3808）ac_auto_machine|counter|trie
P3796（https://www.luogu.com.cn/problem/P3796）ac_auto_machine|counter|trie
P5357（https://www.luogu.com.cn/problem/P5357）ac_auto_machine|counter|trie

"""

from typing import List

from src.strings.ac_auto_machine.template import AhoCorasick


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_1717(big: str, smalls: List[str]) -> List[List[int]]:
        # AC自动机匹配关键词在文本中出现的位置信息
        auto = AhoCorasick(smalls)
        dct = auto.search_in(big)
        return [dct.get(w, []) for w in smalls]