from __future__ import division

import unittest
from typing import List


class Solution:
    def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
        dct = set("aeiou")
        n = len(words)
        pre = [0]*(n+1)
        for i in range(n):
            cur = words[i][0] in dct and words[i][-1] in dct
            pre[i+1] = pre[i]+cur
        return [pre[y+1]-pre[x] for x, y in queries]





class TestGeneral(unittest.TestCase):
    def test_solution(self):
        assert Solution().smallestValue(4) == 4
        return


if __name__ == '__main__':
    unittest.main()
