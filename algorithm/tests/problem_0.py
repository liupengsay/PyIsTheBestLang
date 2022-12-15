import sys
sys.setrecursionlimit(10000000)


def read():
    return sys.stdin.readline()


def ac(x):
    return sys.stdout.write(str(x) + '\n')


def main():
    n, m = map(int, read().split())
    repair = list(map(int, read().split()))

    # 设置初始值距离
    dis = [[float("inf")]*n for _ in range(n)]
    edges = []
    for _ in range(m):
        a, b, c = map(int, read().split())
        dis[a][b] = dis[b][a] = c
        edges.append([a, b, c])
    for i in range(n):
        dis[i][i] = 0

    # 修复村庄之后用 Floyd算法 更新以该村庄为中转的距离
    k = 0
    q = int(read())
    queris = []
    for _ in range(q):
        x, y, t = map(int, read().split())
        queris.append([x, y, t])
        while k < n and repair[k] <= t:
            for a in range(n):
                for b in range(a+1, n):
                    cur = dis[a][k] + dis[k][b]
                    if dis[a][b] > cur:
                        dis[a][b] = dis[b][a] = cur
            k += 1
        if dis[x][y] < float("inf") and x < k and y < k:
            ac(dis[x][y])
        else:
            ac(-1)
    return


main()
