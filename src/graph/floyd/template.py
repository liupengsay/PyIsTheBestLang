from src.utils.fast_io import inf


class Floyd:
    def __init__(self):
        return

    @staticmethod
    def get_cnt_of_shortest_path(edges, n):  # undirected
        dis = [[inf] * n for _ in range(n)]
        cnt = [[0] * n for _ in range(n)]
        for i in range(n):
            dis[i][i] = 0
            cnt[i][i] = 1
        for x, y, w in edges:
            dis[x][y] = dis[y][x] = w
            cnt[x][y] = cnt[y][x] = 1
        for k in range(n):  # mid point
            for i in range(n):  # start point
                if dis[i][k] == inf or i == k:
                    continue
                for j in range(i + 1, n):  # end point
                    if j == k:
                        continue
                    if dis[i][k] + dis[k][j] < dis[j][i]:
                        dis[i][j] = dis[j][i] = dis[i][k] + dis[k][j]
                        cnt[i][j] = cnt[j][i] = cnt[i][k] * cnt[k][j]
                    elif dis[i][k] + dis[k][j] == dis[j][i]:
                        cnt[i][j] += cnt[i][k] * cnt[k][j]
                        cnt[j][i] += cnt[i][k] * cnt[k][j]
        return cnt, dis

    @staticmethod
    def directed_shortest_path(n):
        # Calculate the shortest path between all point pairs using the Floyd algorithm
        dis = [inf] * n * n  # need to be initial
        for k in range(n):  # mid point
            for i in range(n):  # start point
                if dis[i * n + k] == inf:
                    continue
                for j in range(n):  # end point
                    dis[i * n + j] = min(dis[i * n + j], dis[i * n + k] + dis[k * n + j])
        return dis

    @staticmethod
    def undirected_shortest_path(n):
        # Calculate the shortest path between all point pairs using the Floyd algorithm
        dis = [inf] * n * n  # need to be initial
        for k in range(n):  # mid point
            for i in range(n):  # start point
                if dis[i * n + k] == inf:
                    continue
                for j in range(i + 1, n):  # end point
                    dis[j * n + i] = dis[i * n + j] = min(dis[i * n + j], dis[i * n + k] + dis[k * n + j])
        return dis

    @staticmethod
    def undirected_shortest_path_detail(n):
        # Calculate the shortest path between all point pairs using the Floyd algorithm
        dis = [inf] * n * n  # need to be initial
        for k in range(n):  # mid point
            for i in range(n):  # start point
                if dis[i * n + k] == inf:
                    continue
                for j in range(i + 1, n):  # end point
                    dis[j * n + i] = dis[i * n + j] = min(dis[i * n + j], dis[i * n + k] + dis[k * n + j])

        path = []
        for i in range(n):
            if dis[0 * n + i] + dis[i * n + n - 1] == dis[0 * n + n - 1]:
                path.append(i)
        return dis, path
