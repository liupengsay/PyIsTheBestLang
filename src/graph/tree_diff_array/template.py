
class TreeDiffArray:

    def __init__(self):
        # node and edge differential method on tree
        return

    @staticmethod
    def bfs_iteration(dct, queries, root=0):
        """node differential method"""
        n = len(dct)
        stack = [root]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        diff = [0] * n
        for u, v, ancestor in queries:
            # update on the path u to ancestor and v to ancestor
            diff[u] += 1
            diff[v] += 1
            diff[ancestor] -= 1
            if parent[ancestor] != -1:
                diff[parent[ancestor]] -= 1

        # differential summation from bottom to top
        stack = [root]
        while stack:
            i = stack.pop()
            if i >= 0:
                stack.append(~i)
                for j in dct[i]:
                    if j != parent[i]:
                        stack.append(j)
            else:
                i = ~i
                for j in dct[i]:
                    if j != parent[i]:
                        diff[i] += diff[j]
        return diff

    @staticmethod
    def bfs_iteration_edge(dct, queries, root=0):
        # Differential calculation of edges on the tree
        # where the count of edge is dropped to the corresponding down node
        n = len(dct)
        stack = [root]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        # Perform edge difference counting
        diff = [0] * n
        for u, v, ancestor in queries:
            # update the edge on the path u to ancestor and v to ancestor
            diff[u] += 1
            diff[v] += 1
            # make the down node represent the edge count
            diff[ancestor] -= 2

        # differential summation from bottom to top
        stack = [[root, 1]]
        while stack:
            i, state = stack.pop()
            if state:
                stack.append([i, 0])
                for j in dct[i]:
                    if j != parent[i]:
                        stack.append([j, 1])
            else:
                for j in dct[i]:
                    if j != parent[i]:
                        diff[i] += diff[j]
        return diff

    @staticmethod
    def dfs_recursion(dct, queries, root=0):
        n = len(dct)

        stack = [root]
        parent = [-1] * n
        while stack:
            i = stack.pop()
            for j in dct[i]:
                if j != parent[i]:
                    stack.append(j)
                    parent[j] = i

        diff = [0] * n
        for u, v, ancestor in queries:
            diff[u] += 1
            diff[v] += 1
            diff[ancestor] -= 1
            if parent[ancestor] != -1:
                diff[parent[ancestor]] -= 1

        def dfs(x, fa):
            for y in dct[x]:
                if y != fa:
                    diff[x] += dfs(y, x)
            return diff[x]

        dfs(0, -1)
        return diff
