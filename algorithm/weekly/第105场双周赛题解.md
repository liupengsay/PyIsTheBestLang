

***

# [【儿须成名酒须醉】😺 第 105 场力扣夜喵双周赛题解]

***

### 写在前面
掉分不过是一切从头再来，终有破晓之日！希望不被rejudge刷新历史最高分数~

[【儿须成名酒须醉】😺 第 105 场力扣夜喵双周赛题解]: https://leetcode.cn/contest/biweekly-contest-105/
***    
## [题目一: 购买两块巧克力]
[题目一: 购买两块巧克力]: https://leetcode.cn/contest/biweekly-contest-105/problems/buy-two-chocolates/
【儿须成名酒须醉】Python3+贪心
### 解题思路
按照题意取最小花费的两块巧克力计算，超出花费则直接返回本金。
- 贪心
### 代码
```python
class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        a, b = prices[0], prices[1]
        if a > b:
            a, b = b, a
        # 经典的O(n)取出数组最小或者最大两个值的写法，也常用于判断数组每个数除当前数之外最小或者最大的值
        for c in prices[2:]:
            if c < a:
                a, b = c, a
            elif c < b:
                b = c
        return money if a + b > money else money - a - b
```
### 复杂度分析
设数组长度为$n$，则有
- 时间复杂度$O(n)$
- 空间复杂度$O(1)$

***

## [题目二：字符串中的额外字符]

[题目二：字符串中的额外字符]: https://leetcode.cn/contest/biweekly-contest-105/problems/extra-characters-in-a-string/
【儿须成名酒须醉】Python3+集合+动态规划
### 解题思路
由于数据范围比较有限，直接使用集合存储单词集$dct$，使用动态规划记录最优分割，设$dp[i+1]$为$s[:i]$的最优分割结果，则有
$$dp[i+1]=min(dp[i]+1, \min_{0<=k<=i}^{s[k:i+1]\ in\ dct}dp[k])$$
- 动态规划
### 代码
```python
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dct = set(dictionary)
        dp = list(range(n + 1))
        dp[0] = 0
        for i in range(n):
            dp[i + 1] = dp[i] + 1
            for j in range(i + 1):
                if s[j:i + 1] in dct and dp[j] < dp[i + 1]:
                    dp[i + 1] = dp[j]
        return dp[-1]
```
### 复杂度分析
设字符串长度为$n$，单词个数与平均长度为$m$和$k$，则有
- 时间复杂度$O(n+mk)$
- 空间复杂度$O(n^2)$


***
## [题目三：一个小组的最大实力值]

[题目三：一个小组的最大实力值]: https://leetcode.cn/contest/biweekly-contest-105/problems/maximum-strength-of-a-group/
【儿须成名酒须醉】Python3+子集+枚举
### 解题思路
数据长度与数据的取值范围都很有限，因此采用枚举子集的办法计算最大值，这里实现状压枚举与回溯枚举。
- 子集
- 枚举
### 状压
```python
class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        ans = -inf
        n = len(nums)
        for i in range(1, 1<<n):  # 注意不能为空集
            lst = [nums[j] for j in range(n) if i & (1<<j)]
            cur = reduce(mul, lst)
            ans = ans if ans > cur else cur
        return ans
```
### 回溯
```python
class Solution:
    def maxStrength(self, nums: List[int]) -> int:

        def dfs(i):
            nonlocal ans, pre, cnt
            if cnt:  # 注意不能为空集
                ans = ans if ans > pre else pre
            if i == n:
                return
            tmp = pre
            pre *= nums[i]
            cnt += 1
            dfs(i + 1)
            cnt -= 1
            pre = tmp
            dfs(i + 1)
            return

        cnt = 0
        pre = 1
        ans = -inf
        n = len(nums)
        dfs(0)
        return ans
```
进一步可以使用分类讨论，分为正负数和零，贪心选取最大乘积。
### 贪心
```python
class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        pos = [num for num in nums if num > 0]
        neg = [num for num in nums if num < 0]
        zero = [num for num in nums if num == 0]
        if pos:
            ans = reduce(mul, pos)
            if neg:
                floor = max(neg)
                ans *= reduce(mul, neg)
                if len(neg) % 2:
                    ans //= floor
        else:
            ans = 0 if zero else -inf
            m = len(neg)
            if m % 2:
                if m == 1:
                    ans = ans if ans > neg[0] else neg[0]
                else:
                    floor = max(neg)
                    ans = reduce(mul, neg)
                    ans //= floor
            elif m:
                ans = reduce(mul, neg)
        return ans
```

### 复杂度分析
设数组的长度为$n$，则有
- 状压时间复杂度$O(n*2^n)$，回溯时间复杂度为$O(2^n)$，贪心复杂度为$O(n)$
- 空间复杂度$O(n)$

***
## [题目四：最大公约数遍历]

[题目四：最大公约数遍历]: https://leetcode.cn/contest/biweekly-contest-105/problems/greatest-common-divisor-traversal/
【儿须成名酒须醉】Python3+数论+并查集
### 解题思路
任意两个不同下标连通的条件是二者最大公约数大于$1$，且数据范围为$10^5$，考虑使用质因数分解和并查集合并具有相同质因子的下标，任意下标对都可以遍历的充要条件则是并查集整个连通块数为$1$。
- 模拟
- 有序列表

### 代码
```python
class NumberTheoryPrimeFactor:
    def __init__(self, ceil):
        self.ceil = ceil
        self.prime_factor = [[] for _ in range(self.ceil + 1)]
        self.min_prime = [0] * (self.ceil + 1)
        self.get_min_prime_and_prime_factor()
        return

    def get_min_prime_and_prime_factor(self):
        # 模板：计算 1 到 self.ceil 所有数字的最小质数因子
        for i in range(2, self.ceil + 1):
            if not self.min_prime[i]:
                self.min_prime[i] = i
                for j in range(i * i, self.ceil + 1, i):
                    self.min_prime[j] = i

        # 模板：计算 1 到 self.ceil 所有数字的质数分解（可选）
        for num in range(2, self.ceil + 1):
            i = num
            while num > 1:
                p = self.min_prime[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                self.prime_factor[i].append([p, cnt])
        return


class UnionFind:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n
        return

    def find(self, x):
        lst = []
        while x != self.root[x]:
            lst.append(x)
            # 在查询的时候合并到顺带直接根节点
            x = self.root[x]
        for w in lst:
            self.root[w] = x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.size[root_x] >= self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # 将非根节点的秩赋0
        self.size[root_x] = 0
        self.part -= 1
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        # 获取每个根节点对应的组
        part = defaultdict(list)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        # 获取每个根节点对应的组大小
        size = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
        return size


# 预处理10^5以内每个数字的质因数分解
nt = NumberTheoryPrimeFactor(10**5)



class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)

        # 将索引归类到对应的质数因子
        ind = defaultdict(list)
        for i in range(n):
            for num, _ in nt.prime_factor[nums[i]]:
                ind[num].append(i)
        
        # 合并同一个质数因子的连通块
        uf = UnionFind(n)
        for num in ind:
            for i in ind[num]:
                uf.union(ind[num][0], i)
        return uf.part == 1
```

由于这题不需要具体的每个数的质因数分解，只需要知道质数因子，因此可以使用下面的算法计算质因数。

```python

def get_num_prime_factor(ceil):
    # 模板：快速计算 1~ceil 的所有质数因子
    prime = [[] for _ in range(ceil + 1)]
    for i in range(2, ceil + 1):
        if not prime[i]:
            prime[i].append(i)
            # 从 i*i 开始作为 prime[j] 的最小质数因子
            for j in range(i * 2, ceil + 1, i):
                prime[j].append(i)
    return prime


prime_factor = get_num_prime_factor(10**5)

```

### 复杂度分析
设数组最大值为$m$，即$10^5$，长度为$n$，则有
- 预处理时间复杂度$O(mlogm)$，遍历计算数组连通块上限为$O(nlogn)$
- 预处理时间复杂度$O(m)$，遍历计算数组连通块上限为$O(n)$
***

### 写在最后
谢谢阅读，继续努力，如有错漏，敬请指正！
***
