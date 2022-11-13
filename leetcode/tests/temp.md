***
### 解题思路
【儿须成名酒须醉】Python3+最短路Dijkstra

### 代码
- 执行用时：328 ms, 在所有 Python3 提交中击败了 30.28% 的用户
- 内存消耗：16.9 MB, 在所有 Python3 提交中击败了 30.89% 的用户
- 通过测试用例：42 / 42


```python3
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: [-x[0], x[1]])
        ans = []
        for p in people:
            ans.insert(p[1], p)
        return ans
```

***
### 解题思路
【儿须成名酒须醉】Python3+记忆化搜索

### 代码
- 执行用时：100 ms, 在所有 Python3 提交中击败了 88.32% 的用户
- 内存消耗：22.2 MB, 在所有 Python3 提交中击败了 65.94% 的用户
- 通过测试用例：52 / 52

```python3
class Solution:
    def canCross(self, stones: List[int]) -> bool:

        @lru_cache(None)
        def dfs(k, i):
            # 搜索[跳数k, 位置i]
            if i == n - 1:
                return True
            res = False
            for x in [k - 1, k, k + 1]:
                if x > 0 and stones[i] + x in ind:
                    res = res or dfs(x, ind[stones[i] + x])
            return res

        ind = {num: i for i, num in enumerate(stones)}
        n = len(stones)
        if stones[1] != 1:
            return False
        return dfs(1, 1)
```

[null, null, null, "branford", null, "alps", null, "bradford", null, "bradford", null, "bradford", "orland"]
[null,null,null,"branford",null,"alps",null,"bradford",null,"bradford",null,"bradford","orland"]