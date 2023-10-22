class TopologicalSort:
    def __init__(self):
        return

    @staticmethod
    def get_rank(n, edges):
        dct = [list() for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            degree[j] += 1
            dct[i].append(j)
        stack = [i for i in range(n) if not degree[i]]
        visit = [-1] * n
        step = 0
        while stack:
            for i in stack:
                visit[i] = step
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
            step += 1
        return visit

    @staticmethod
    def count_dag_path(n, edges):
        # Calculate the number of paths in a directed acyclic connected graph
        edge = [[] for _ in range(n)]
        degree = [0] * n
        for i, j in edges:
            edge[i].append(j)
            degree[j] += 1
        cnt = [0] * n
        stack = [i for i in range(n) if not degree[i]]
        for x in stack:
            cnt[x] = 1
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    cnt[j] += cnt[i]
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return cnt

    @staticmethod
    def is_topology_unique(dct, degree, n):
        # Determine whether it is unique while ensuring the existence of topological sorting
        ans = []
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            ans.extend(stack)
            if len(stack) > 1:
                return False
            nex = []
            for i in stack:
                for j in dct[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return True

    @staticmethod
    def is_topology_loop(edge, degree, n):
        # using Topological Sorting to Determine the Existence of Rings in a Directed Graph
        stack = [i for i in range(n) if not degree[i]]
        while stack:
            nex = []
            for i in stack:
                for j in edge[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        nex.append(j)
            stack = nex
        return all(x == 0 for x in degree)

    @staticmethod
    def bfs_topologic_order(n, dct, degree):
        # topological sorting determines whether there are rings in a directed graph
        # while recording the topological order of nodes
        order = [0] * n
        stack = [i for i in range(n) if degree[i] == 0]
        ind = 0
        while stack:
            nex = []
            for i in stack:
                order[i] = ind
                ind += 1
                for j in dct[i]:
                    degree[j] -= 1
                    if degree[j] == 0:
                        nex.append(j)
            stack = nex[:]
        if any(d > 0 for d in degree):
            return []
        return order


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
