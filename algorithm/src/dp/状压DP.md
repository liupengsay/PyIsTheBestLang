

### 通过深搜回溯枚举全部子集的办法比位运算枚举效率更高
参考题目：
https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/solution/by-liupengsay-6dpl/


```python3
# https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/

IND = {1 << i: i for i in range(15)}


class Solution:
    def minSessions(self, tasks: List[int], sessionTime: int) -> int:
        n = len(tasks)
        group = set()
        for i in range(1, 1 << n):
            cur = 0
            sub = i
            while sub:
                cur += tasks[IND[sub & (-sub)]]
                sub &= (sub - 1)
            if cur <= sessionTime:
                group.add(i)
        dp = [n] * (1 << n)
        dp[0] = 0
        for i in range(1, 1 << n):
            sub = i
            while sub:
                if dp[i^sub] + dp[sub] < dp[i]:
                    dp[i] = dp[i^sub] + dp[sub]
                sub &=(sub-1)
        return dp[-1]
```



```python3
from collections import defaultdict

class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        dct = defaultdict(int)
        for x, y, z in transactions:
            dct[x] += z
            dct[y] -= z
        nums = [dct[k] for k in dct if dct[k]]
        n = len(nums)

        group = dict()
        # 枚举子集为0
        for i in range(1<<n):
            amount = 0
            for j in range(n):
                if i&(1<<j):
                    amount += nums[j]
            if not amount:
                group[i] = 1

        # 状态压缩
        dp = dict()
        dp[0] = 0
        for i in range(1<<n):
            sub = i
            while sub>0:
                if sub in group and sub^i in dp:
                    if i not in dp:
                        dp[i] = dp[sub^i] + 1
                    else:
                        dp[i] = max(dp[i], dp[sub^i]+1)
                sub = (sub-1)&i
        return n-dp[(1<<n)-1]

```

```python3
sub = [[] for _ in range(1<<12)]
sec = [[] for _ in range(1<<12)]
for i in range(1 << 12):
    s = i
    while s > 0:
        s = (s - 1) & i
        sub[i].append([s, i^s])
    sec[i].extend([i&-i, i^(i&-i)])

class Solution:
    def minimumTimeRequired(self, jobs, k: int) -> int:
        # 状态压缩初始化
        n = len(jobs)
        m = dp[n]
        state = [0]*m
        for i in range(n):
            state[dp[i]] = jobs[-i-1]
        for i in range(m):
            state[i] = state[sec[i][0]] + state[sec[i][1]]

        # 进行状态迭代
        cur = state[:]
        for j in range(1, k):
            for i in range(m-1, -1, -1):
                for s in sub[i]:
                    candidate = state[s[0]]
                    if cur[s[1]] > candidate:
                        candidate = cur[s[1]]
                    if candidate < cur[i]:
                        cur[i] = candidate
        return cur[-1]

作者：liupengsay
链接：https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/solution/er-xu-cheng-ming-jiu-xu-zui-python3zhuan-rtfb/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```