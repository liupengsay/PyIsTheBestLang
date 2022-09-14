



【儿须成名酒须醉】Python3+状态压缩
- 执行用时： 36 ms , 在所有 Python3 提交中击败了 81.48% 的用户
- 内存消耗： 14.9 MB , 在所有 Python3 提交中击败了 74.82% 的用户
- 通过测试用例： 80 / 80

芒果台开关游戏：这是开（操作1为奇数次，偶数次为关）、这是开（操作2为奇数次，偶数次为关） 、这是开（操作3为奇数次，偶数次为关）、这是开（操作4为奇数次，偶数次为关），这是开还是关？
每个操作偶数次不改变灯泡状态，因为可以枚举每种操作的奇偶次数组合，计算相应的灯泡序列状态，想不到O(1)的复杂度，暴力比较存储O(n)也可
- 执行用时： 40 ms, 在所有 Python3 提交中击败了 91.64 % 的用户
- 内存消耗： 17.8 MB, 在所有 Python3 提交中击败了 17.48 % 的用户
- 通过测试用例： 86 / 86
```python3

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
