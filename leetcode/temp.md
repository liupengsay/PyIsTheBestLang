

【儿须成名酒须醉】Python3+堆（优先队列）
- 执行用时： 40 ms , 在所有 Python3 提交中击败了 91.64% 的用户 
- 内存消耗： 17.8 MB , 在所有 Python3 提交中击败了 17.48% 的用户
- 通过测试用例： 86 / 86
```python3

```

# 方法二：前缀和
# 提交结果
- 执行用时： 60 ms , 在所有 Python3 提交中击败了 14.71% 的用户
- 内存消耗： 15.8 MB , 在所有 Python3 提交中击败了 10.29% 的用户
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
