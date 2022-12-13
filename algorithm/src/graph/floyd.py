
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        d = [[n + 1] * n for _ in range(n)]
        for i in range(n):
            for j in graph[i]:
                d[i][j] = 1
        
        # 使用 floyd 算法预处理出所有点对之间的最短路径长度
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])
        return dp