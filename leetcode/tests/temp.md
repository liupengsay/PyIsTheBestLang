***
### 解题思路
【儿须成名酒须醉】Python3+暴力

### 代码
- 执行用时：112 ms, 在所有 Python3 提交中击败了 33.89% 的用户
- 内存消耗：15 MB, 在所有 Python3 提交中击败了 47.78% 的用户
- 通过测试用例：73 / 73

```python3
class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        ceil = floor = pre = 0
        for num in nums:
            pre += num
            ceil = ceil if ceil > pre else pre
            floor = floor if floor < pre else pre
        return ceil - floor
```


***
### 解题思路
【儿须成名酒须醉】Python3+贪心+广度优先搜索

### 代码
- 执行用时：1144 ms, 在所有 Python3 提交中击败了 89.94% 的用户
- 内存消耗：49.9 MB, 在所有 Python3 提交中击败了 93.30% 的用户
- 通过测试用例：59 / 59

```python3
import random

P = random.randint(26, 100)
MOD = random.randint(10 ** 9 + 7, 2 ** 31 - 1)

DP = [1]
for _ in range(1,  2001):
    DP.append((DP[-1] * P) % MOD)


class Solution:
    def distinctEchoSubstrings(self, text: str) -> int:
        n = len(text)
        ans = 0
        for k in range(1, n // 2 + 1):
            pre = [() for _ in range(n)]
            dup = set()
            
            num = 0
            for i in range(k):
                num += DP[k - 1 - i] * (ord(text[i]) - ord('a'))
                num %= MOD
            pre[i] = num

            for i in range(k, n):
                num -= DP[k - 1] * (ord(text[i - k]) - ord('a'))
                num *= P
                num += DP[0] * (ord(text[i]) - ord('a'))
                num %= MOD

                pre[i] = num
                if pre[i] == pre[i-k]:
                    dup.add(num)
            ans += len(dup)
        return ans
```
