
### 解题思路
【儿须成名酒须醉】Python3+状压DP

### 代码
- 执行用时：876 ms, 在所有 Python3 提交中击败了 50.75% 的用户
- 内存消耗：15.2 MB, 在所有 Python3 提交中击败了 74.63% 的用户
- 通过测试用例：84 / 84
```python3

def get_k_bin_of_n(n: int, k: int, m: int) -> int:
    lst = []
    while n:
        lst.append(n % k)
        n //= k
    lst = lst + [0] * (m - len(lst))
    return lst


class Solution:
    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        n = len(nums)
        dp = [0] * (3 ** numSlots)
        for sub in range(3 ** numSlots):
            cnt = get_k_bin_of_n(sub, 3, numSlots)
            pre = sum(cnt)
            if pre >= n:
                continue
            for j in range(numSlots):
                if cnt[j] < 2:
                    cur = dp[sub] + (nums[pre] & (j + 1))
                    dp[sub + 3**j] = max(dp[sub + 3**j], cur)
        return max(dp)
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
