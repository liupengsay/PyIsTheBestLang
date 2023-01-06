***
### 解题思路
【儿须成名酒须醉】Python3+递归+二叉树+中缀表达式
- 相似题目[P1175 表达式的转换](https://www.luogu.com.cn/problem/P1175)

### 代码
- 执行用时：44 ms, 在所有 Python3 提交中击败了 22.73% 的用户
- 内存消耗：15.2 MB, 在所有 Python3 提交中击败了 9.09% 的用户
- 通过测试用例：64 / 64

```python3
class Node(object):
    def __init__(self, val=" ", left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def expTree(self, s: str) -> 'Node':
        # 只剩数字的情况
        if s.isnumeric():
            return Node(s)

        # 不支持 2*-3 和 -2+3 的情形即要求所有数字为非负数
        n = len(s)
        cnt = 0

        # 按照运算符号的优先级倒序遍历字符串
        for i in range(n - 1, -1, -1):
            cnt += int(s[i] == ')') - int(s[i] == '(')
            if s[i] in ['+', '-'] and not cnt:
                return Node(s[i], self.expTree(s[:i]), self.expTree(s[i + 1:]))

        # 注意是从后往前
        for i in range(n - 1, -1, -1):
            cnt += int(s[i] == ')') - int(s[i] == '(')
            if s[i] in ['*', '/'] and not cnt:
                return Node(s[i], self.expTree(s[:i]), self.expTree(s[i + 1:]))

        # 注意是从前往后
        for i in range(n):
            cnt += int(s[i] == ')') - int(s[i] == '(')
            if s[i] in ['^'] and not cnt:  # 这里的 ^ 表示幂
                return Node(s[i], self.expTree(s[:i]), self.expTree(s[i + 1:]))

        # 其余则是开头结尾为括号的情况
        return self.expTree(s[1:-1])
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

昨晚掉大分，今天就喜提$rank1$，这就是周赛的魅力吧！就像开盲盒，打开之前永远不知道里面有什么~