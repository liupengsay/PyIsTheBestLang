***
### 解题思路
【儿须成名酒须醉】Python3+哈希计数+栈
- 参考官解
### 代码
- 执行用时：232 ms, 在所有 Python3 提交中击败了 100.00% 的用户
- 内存消耗：22.9 MB, 在所有 Python3 提交中击败了 40.59% 的用户
- 通过测试用例：38 / 38

```python3
class FreqStack:

    def __init__(self):
        self.dct = defaultdict(list)
        self.freq = 0
        self.cnt = defaultdict(int)

    def push(self, val: int) -> None:
        self.cnt[val] += 1
        self.dct[self.cnt[val]].append(val)
        self.freq = self.freq if self.freq > self.cnt[val] else self.cnt[val]

    def pop(self) -> int:
        val = self.dct[self.freq].pop()
        self.cnt[val] -= 1
        if not self.dct[self.freq]:
            self.freq -= 1
        return val
```

***
### 解题思路
【儿须成名酒须醉】Python3+贪心
- 参考大佬题解
### 代码
- 执行用时：56 ms, 在所有 Python3 提交中击败了 73.75% 的用户
- 内存消耗：16.1 MB, 在所有 Python3 提交中击败了 76.25% 的用户
- 通过测试用例：117 / 117


```python3
class Solution:
    def intersectionSizeTwo(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda inter: [inter[1], -inter[0]])
        ans, a, b = 0, -1, -1
        for x, y in intervals:
            if x > b:
                ans, a, b = ans + 2, y - 1, y
            elif x > a:
                ans, a, b = ans + 1, b, y
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