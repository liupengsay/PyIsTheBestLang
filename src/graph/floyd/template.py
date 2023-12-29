from src.utils.fast_io import inf


class Floyd:
    def __init__(self):
        return

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
