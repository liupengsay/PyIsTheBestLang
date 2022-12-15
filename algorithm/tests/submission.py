import sys
input = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(str(x)+'\n')
sys.setrecursionlimit(10000000)


def main():
    n, m, b = map(int, input().split())
    cost = [0]*n
    for i in range(n):
        cost[i] = int(input())

    dct = [dict() for _ in range(n)]
    for _ in range(m):
        a, b, c = map(int, input().split())
        a -= 1
        b -= 1
        if b not in dct[a] or dct[a][b] > c:
            dct[a][b] = c
        if a not in dct[b] or dct[b][a] > c:
            dct[b][a] = c

    def check():
        visit = [float("inf")]*n
        blood = [float("-inf")]*n
        stack = [[cost[0], 0, b]]
        while stack:
            dis, i, bd = heapq.heappop(stack)
            if i == n-1:
                return str(dis)
            if dis >= visit[i] and bd <= blood[i]:
                continue
            if dis<visit[i]:
                visit[i] = dis
                blood[i] = bd
            for j in dct[i]:
                heapq.heappush(stack, [max(dis, cost[j]), j, bd-dct[i][j]])
        return "AFK"

    print(check())
    return

main()
