
### 解题思路
【儿须成名酒须醉】Python3+堆（优先队列）

### 代码
- 执行用时：36 ms, 在所有 Python3 提交中击败了 99.90% 的用户
- 内存消耗：15.2 MB, 在所有 Python3 提交中击败了 17.23% 的用户
- 通过测试用例：596 / 596
```python3
dp = []
stack = [1]
visit = set([1])
while len(dp) < 1690:
    num = heapq.heappop(stack)
    dp.append(num)
    for p in [2, 3, 5]:
        if num*p not in visit:
            visit.add(num*p)
            heapq.heappush(stack, num*p)


class Solution:
    def nthUglyNumber(self, k: int) -> int:
        return dp[k-1]
```

[1820. 最多邀请的个数]: https://leetcode.cn/problems/maximum-number-of-accepted-invitations/solution/er-xu-cheng-ming-jiu-xu-zui-python3xiong-ikt7/


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
