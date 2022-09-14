
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict

# class Solution:
#     def earliestAndLatest(self, n: int, firstPlayer: int, secondPlayer: int) -> List[int]:
#
#         def dfs(tup, step):
#             nonlocal small, big
#             if tup in visit:
#                 return
#             visit.add(tup)
#
#             i = 0
#             j = len(tup)-1
#             stack = [[]]
#             while i<j:
#                 nex = []
#                 if tup[i] == firstPlayer and tup[j] == secondPlayer:
#                     small = min(small, step)
#                     big = max(big, step)
#                     return
#                 for path in stack:
#                     nex.append(path+[tup[i]])
#                     nex.append(path+[tup[j]])
#                 i += 1
#                 j -= 1
#                 stack = nex[:]
#
#
#             for state in stack:
#                 if i == j:
#                     state.append(tup[i])
#                 dfs(tuple(sorted(state)), step+1)
#             return
#
#         visit = set()
#         small = n
#         big = 0
#         dfs(tuple(list(range(1, n+1))), 1)
#         return [small, big]

@lru_cache(None)
def check(i, j):
    if i > j:
        return [[]]
    if i == j:
        return [[i]]
    res = []
    for nex in check(i+1, j-1):
        res.append([i]+nex)
        res.append(nex+[j])
    return res


print(len(check(0, 27)))