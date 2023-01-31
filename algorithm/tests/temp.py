import io, os
import collections
input = io.BytesIO(os.read(0,os.fstat(0).st_size)).readline
R=lambda:map(int,input().split())

n,m=R()
graph=[[2222244444]*(n+1) for _ in range(n+1)]
for _ in range(m):
    u,v,w=R()
    graph[u][v]=graph[v][u]=w
for i in range(n+1):
    graph[i][i]=0
for k in range(1,n+1):
    for i in range(1,n+1):
        for j in range(1,n+1):
            if graph[i][j]>graph[i][k]+graph[k][j]:
                graph[i][j]=graph[j][i]=graph[i][k]+graph[k][j]
res=2222244444
for i in range(1,n):
    for j in range(i+1,n+1):
        cur=0
        for ii in range(1,n):
            for jj in range(ii+1,n+1):
                cur+=min(graph[ii][jj],graph[ii][i]+graph[j][jj],graph[ii][j]+graph[i][jj])
        res=min(cur,res)
print(res)