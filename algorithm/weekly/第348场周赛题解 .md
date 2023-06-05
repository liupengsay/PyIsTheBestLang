

***

# [【儿须成名酒须醉】第 347 场力扣周赛题解]

***

### 竞赛日记
苦海无边，回头是岸！

[【儿须成名酒须醉】第 347 场力扣周赛题解]: https://leetcode.cn/contest/weekly-contest-347/

***    
## [题目一: 移除字符串中的尾随零]
[题目一: 移除字符串中的尾随零]: https://leetcode.cn/contest/weekly-contest-326/problems/count-the-digits-that-divide-a-number/
【儿须成名酒须醉】Python3+字符串
### 解题思路
正整数，没有前导零，直接调用API即可。
- 字符串
### 代码
```python3
class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        return num.rstrip("0")
```
### 复杂度分析
设字符串长度为$n$，则有
- 时间复杂度$O(n)$
- 空间复杂度$O(n)$

***
## [题目二：对角线上不同值的数量差]

[题目二：对角线上不同值的数量差]: https://leetcode.cn/contest/weekly-contest-347/problems/difference-of-number-of-distinct-values-on-diagonals/
【儿须成名酒须醉】Python3+模拟计数
### 解题思路
预处里每个位置左上角和右下角的不同值个数。
- 计数
- 模拟

### 代码
```python3
class Solution:
    def differenceOfDistinctValues(self, grid: List[List[int]]) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        ans = [[0] * n for _ in range(m)]

        pre = [[0] * n for _ in range(m)]
        for j in range(n):
            dct = {grid[0][j]}
            x, y = 1, j + 1
            while 0 <= x < m and 0 <= y < n:
                pre[x][y] = len(dct)
                dct.add(grid[x][y])
                x += 1
                y += 1
        for i in range(1, m):
            dct = {grid[i][0]}
            x, y = i + 1, 1
            while 0 <= x < m and 0 <= y < n:
                pre[x][y] = len(dct)
                dct.add(grid[x][y])
                x += 1
                y += 1

        post = [[0] * n for _ in range(m)]
        for j in range(n):
            dct = {grid[m - 1][j]}
            x, y = m - 2, j - 1
            while 0 <= x < m and 0 <= y < n:
                post[x][y] = len(dct)
                dct.add(grid[x][y])
                x -= 1
                y -= 1
        for i in range(m - 1):
            dct = {grid[i][n - 1]}
            x, y = i - 1, n - 2
            while 0 <= x < m and 0 <= y < n:
                post[x][y] = len(dct)
                dct.add(grid[x][y])
                x -= 1
                y -= 1
        for i in range(m):
            for j in range(n):
                ans[i][j] = abs(pre[i][j] - post[i][j])
        return ans
```
### 复杂度分析
设矩阵的大小为$m$和$n$，则有
- 时间复杂度$O(mn)$
- 空间复杂度$O(mn)$


***
## [题目三：使所有字符相等的最小成本]

[题目三：使所有字符相等的最小成本]: https://leetcode.cn/contest/weekly-contest-347/problems/minimum-cost-to-make-all-characters-equal/
【儿须成名酒须醉】Python3+动态规划
### 解题思路
正反两遍遍历字符串，记录$dp[i][0]$和$dp[i][1]$为全部变为$0$或$1$的最小代价。
- 动态规划

### 代码
```python3
class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        dp1 = [[0, 0] for _ in range(n + 1)]
        for i, w in enumerate(s):
            if w == "1":
                dp1[i + 1][1] = min(dp1[i][1], dp1[i][0] + i)
                dp1[i + 1][0] = min(dp1[i][1] + i + 1, dp1[i][0] + i + 1 + i)
            elif w == "0":
                dp1[i + 1][1] = min(dp1[i][1] + i + 1 + i, dp1[i][0] + i + 1)
                dp1[i + 1][0] = min(dp1[i][0], dp1[i][1] + i)

        # 等同于前后缀
        s = s[::-1]
        dp2 = [[0, 0] for _ in range(n + 1)]
        for i, w in enumerate(s):
            if w == "1":
                dp2[i + 1][1] = min(dp2[i][1], dp2[i][0] + i)
                dp2[i + 1][0] = min(dp2[i][1] + i + 1, dp2[i][0] + i + 1 + i)
            elif w == "0":
                dp2[i + 1][1] = min(dp2[i][1] + i + 1 + i, dp2[i][0] + i + 1)
                dp2[i + 1][0] = min(dp2[i][0], dp2[i][1] + i)
        dp2 = dp2[::-1]
        ans = min(min(dp1[i][0] + dp2[i][0], dp1[i][1] + dp2[i][1]) for i in range(n + 1))
        return ans
```
### 复杂度分析
设字符串长度为$n$，则有
- 时间复杂度$O(n)$
- 空间复杂度$O(n)$

***
## [题目四：矩阵中严格递增的单元格数]

[题目四：矩阵中严格递增的单元格数]: https://leetcode.cn/contest/weekly-contest-347/problems/maximum-strictly-increasing-cells-in-a-matrix/
【儿须成名酒须醉】Python3+排序+动态规划
### 解题思路
将元素值从小到大排序，由于是严格递增，因此可以使用动态规划，$row[i]$和$col[j]$分别表示当前行与当前列最长的递增序列长度。
- 排序 
- 动态规划

### 代码
```python3
class Solution:
    def maxIncreasingCells(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        dct = defaultdict(list)
        for i in range(m):
            for j in range(n):
                dct[mat[i][j]].append([i, j])
        row = [0] * m
        col = [0] * n
        for val in sorted(dct):
            lst = []
            for i, j in dct[val]:
                x = row[i] if row[i] > col[j] else col[j]
                lst.append([i, j, x + 1])
            for i, j, w in lst:
                col[j] = col[j] if col[j] > w else w
                row[i] = row[i] if row[i] > w else w
        return max(max(row), max(col))
```


### 复杂度分析
设矩阵的大小为$m$和$n$，则有
- 时间复杂度$O(mnlogmn)$
- 空间复杂度$O(mn)$
***

### 写在最后
谢谢阅读，继续努力！