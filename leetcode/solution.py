import itertools
from typing import List


class Trie:
    def __init__(self):
        self.dct = dict()
        return

    def update(self, word):
        cur = self.dct
        for w in word:
            if w not in cur:
                cur[w] = dict()
            cur = cur[w]
        return


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        target = [0]*26
        for w in p:
            target[ord(w)-ord('a')] += 1

        n = len(p)
        ans = []
        cur = [0]*26
        for i, w in enumerate(s):
            cur[ord(w)-ord('a')] += 1
            if i >= n:
                cur[ord(s[i-n])-ord('a')] += 1
                if cur == target:
                    ans.append(i)
        return ans













[73,55,36,5,55,14,9,7,72,52]
32
69
[-1, 1, 1, 0, 1, 0, 0, 0, -1, 1]