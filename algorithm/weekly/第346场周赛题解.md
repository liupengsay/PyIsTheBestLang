

***

# [【儿须成名酒须醉】第 346 场力扣周赛题解]

***

### 竞赛日记
> 第 342 场周赛，掉分！
> 第 343 场周赛，又掉分！
> 第 344 场周赛，还是掉分！
> 第 345 场周赛，终于上分了

[【儿须成名酒须醉】第 346 场力扣周赛题解]: https://leetcode.cn/contest/weekly-contest-346/


***
## [题目四：修改图中的边权]

[题目四：修改图中的边权]: https://leetcode.cn/contest/weekly-contest-346/problems/modify-graph-edge-weights/

【儿须成名酒须醉】Python3+Dijkstra+贪心+最小生成树思想


### 解题思路
1. 首先注意到target的范围，与边权最大值范围，先将所有的$-1$改为$2*10^{9}$。
2. 计算初始的最短路，如果小于target则无论怎么修改肯定无解，如果等于target则直接返回。
3. 如果大于target，则尝试依次将每条边权$2*10^{9}$改为$1$，计算此时最短路dis，如果此时dis小于等于target，则有解，并将此时这条边权增加target-dis，输出即可。
4. 具体需要注意一些细节.

### 代码
```python
class Dijkstra:
    def __init__(self):
        return

    @staticmethod
    def dijkstra(dct, src):
        # 模板: Dijkstra求起终点的最短路，注意只能是正权值可以提前返回结果，并返回对应经过的路径
        n = len(dct)
        dis = [float("inf")] * n
        stack = [[0, src]]
        dis[src] = 0
        while stack:
            d, i = heapq.heappop(stack)
            if dis[i] < d:
                continue
            for j in dct[i]:
                dj = dct[i][j] + d
                if dj < dis[j]:
                    dis[j] = dj
                    heapq.heappush(stack, [dj, j])
        return dis


class Solution:
    def modifiedGraphEdges(self, n: int, edges: List[List[int]], source: int, destination: int, target: int) -> List[
        List[int]]:
        ceil = 2 * 10 ** 9
        dct = [dict() for _ in range(n)]
        edge = []
        cut = set()
        for i, j, w in edges:
            if w == -1:
                edge.append([i, j, w])
                w = ceil
            dct[i][j] = dct[j][i] = w
            cut.add((i, j))
            cut.add((j, i))

        dis = Dijkstra().dijkstra(dct, source)
        if dis[destination] == target:
            ans = []
            visit = set()
            for i in range(n):
                for j in dct[i]:
                    if (i, j) not in visit:
                        ans.append([i, j, dct[i][j]])
                        visit.add((i, j))
                        visit.add((j, i))
            return ans
        if dis[destination] < target:
            return []

        for x, y, w in edge:
            dct[x][y] = dct[y][x] = 1
            dis = Dijkstra().dijkstra(dct, source)
            if dis[destination] <= target:
                gap = target - dis[destination]
                dct[x][y] = dct[y][x] = dct[x][y] + gap
                ans = []
                visit = set()
                for i in range(n):
                    for j in dct[i]:
                        if (i, j) not in visit:
                            ans.append([i, j, dct[i][j]])
                            visit.add((i, j))
                            visit.add((j, i))
                return ans
        return []
```


### 复杂度分析
设节点个数为$n$，边条数为$m$，则有
- 时间复杂度$O(mn^2logn)$
- 空间复杂度$O(nm)$
***

### 写在最后
谢谢阅读，继续努力，如有错漏，敬请指正！
***
