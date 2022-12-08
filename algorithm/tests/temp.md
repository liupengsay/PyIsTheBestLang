***
### 解题思路
【儿须成名酒须醉】Python3+枚举
- 枚举表达式所有可能的状态计算

### 代码
- 执行用时：1488 ms, 在所有 Python3 提交中击败了 5.15% 的用户
- 内存消耗：15.1 MB, 在所有 Python3 提交中击败了 16.11% 的用户
- 通过测试用例：71 / 71

```python3
class Solution:
    def judgePoint24(self, cards: List[int]) -> bool:

        for a, b, c, d in permutations(cards, 4):
            for op1 in "+-*/":
                for op2 in "+-*/":
                    for op3 in "+-*/":
                        dct = [f"{a}{op1}{b}{op2}{c}{op3}{d}",

                               f"({a}{op1}{b}){op2}{c}{op3}{d}",
                               f"{a}{op1}({b}{op2}{c}){op3}{d}",
                               f"{a}{op1}{b}{op2}({c}{op3}{d})",
                               f"({a}{op1}{b}{op2}{c}){op3}{d}",
                               f"{a}{op1}({b}{op2}{c}{op3}{d})",

                               f"({a}{op1}{b}){op2}({c}{op3}{d})",
                               f"({a}{op1}{b}{op2}{c}){op3}{d}",
                               f"{a}{op1}({b}{op2}({c}{op3}{d}))"]
                        for st in dct:
                            try:
                                if abs(24 - eval(st)) < 1e-5:
                                    return True
                            except BaseException as _:
                                pass
        return False
```

***
### 解题思路
【儿须成名酒须醉】Python3+三指针

### 代码
- 执行用时：696 ms, 在所有 Python3 提交中击败了 29.68% 的用户
- 内存消耗：15 MB, 在所有 Python3 提交中击败了 91.78% 的用户
- 通过测试用例：315 / 315


```python3
class Solution:
    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n - 2):
            j, k = i + 1, n - 1
            while j < k:
                cur = nums[i] + nums[j] + nums[k]
                if cur >= target:
                    k -= 1
                else:
                    # 注意这里的计数与移动方向
                    ans += k - j
                    j += 1
        return ans
```

***
### 解题思路
【儿须成名酒须醉】Python3+差分数组+前缀和

### 代码
- 执行用时：1592 ms, 在所有 Python3 提交中击败了 60.64% 的用户
- 内存消耗：16.5 MB, 在所有 Python3 提交中击败了 42.55% 的用户
- 通过测试用例：97 / 97

```python3
from sortedcontainers import SortedDict

class MyCalendarTwo:

    def __init__(self):
        self.dct = SortedDict()

    def book(self, start: int, end: int) -> bool:
        self.dct[start] = self.dct.get(start, 0) + 1
        self.dct[end] = self.dct.get(end, 0) - 1
        pre = 0
        for k in self.dct:
            pre += self.dct[k]
            if pre >= 3:
                self.dct[start] -= 1
                self.dct[end] += 1
                return False
        return True
```


### 性能用例
```python3
books = [[i, i+100] for i in range(1, 100000)]
```