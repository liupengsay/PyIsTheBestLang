
### 解题思路
【儿须成名酒须醉】Python3+广度优先搜索
### 代码
- 执行用时：1160 ms, 在所有 Python3 提交中击败了 79.29% 的用户
- 内存消耗：15.1 MB, 在所有 Python3 提交中击败了 50.71% 的用户
- 通过测试用例：60 / 60
```python3
class Solution:
    def kSimilarity(self, s1: str, s2: str) -> int:
        n = len(s1)
        lst1, lst2 = list(s1), list(s2)
        visit = {tuple(lst1)}
        step = 0
        stack = [(lst1, 0)]
        while stack:
            nex = []
            for pre, i in stack:
                while i < n and pre[i] == lst2[i]:
                    i += 1
                if i == n:
                    return step
                for j in range(i+1, n):
                    if pre[j] == lst2[i] != lst2[j]:
                        pre[j], pre[i] = pre[i], pre[j]
                        if tuple(pre) not in visit:
                            visit.add(tuple(pre))
                            nex.append([pre[:], i+1])
                        pre[j], pre[i] = pre[i], pre[j]
            stack = nex
            step += 1
        return -1
```

# 方法二：前缀和
# 提交结果
- 执行用时： 60 ms, 在所有 Python3 提交中击败了 14.71 % 的用户
- 内存消耗： 15.8 MB, 在所有 Python3 提交中击败了 10.29 % 的用户
- 通过测试用例： 63 / 63
# 解题思路


# 代码
```python3


class Solution:
    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        m, n = len(mat), len(mat[0])
        pre = SortedList(mat[0])
        for i in range(1, m):
            cur = SortedList()
            for num in mat[i]:
                for p in pre:
                    cur.add(num + p)
                    if len(cur) > k and num + p > cur[k - 1]:
                        break
            pre = cur[:k]
        return pre[k - 1]


```
