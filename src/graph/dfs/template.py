

class DFS:
    def __init__(self):
        return

    @staticmethod
    def gen_bfs_order_iteration(dct, root=0):
        """template of dfs order for rooted tree"""
        n = len(dct)
        for i in range(n):
            # visit from small to large according to the number of child nodes
            dct[i].sort(reverse=True)  # which is not necessary

        order = 0
        # index is original node value is dfs order
        start = [-1] * n
        # index is original node value is the maximum subtree dfs order
        end = [-1] * n
        # index is original node and value is its parent
        parent = [-1] * n
        stack = [root]
        # depth of every original node
        depth = [0] * n
        # index is dfs order and value is original node
        order_to_node = [-1] * n
        while stack:
            i = stack.pop()
            if i >= 0:
                start[i] = order
                order_to_node[order] = i
                end[i] = order
                order += 1
                stack.append(~i)
                for j in dct[i]:
                    # the order of son nodes can be assigned for lexicographical order
                    if j != parent[i]:
                        parent[j] = i
                        depth[j] = depth[i] + 1
                        stack.append(j)
            else:
                i = ~i
                if parent[i] != -1:
                    end[parent[i]] = end[i]

        return start, end


class DfsEulerOrder:
    def __init__(self, dct, root=0):
        """dfs and euler order of rooted tree which can be used for online point update and query subtree sum"""
        n = len(dct)
        for i in range(n):
            # visit from small to large according to the number of child nodes
            dct[i].sort(reverse=True)  # which is not necessary
        # index is original node value is dfs order
        self.start = [-1] * n
        # index is original node value is the maximum subtree dfs order
        self.end = [-1] * n
        # index is original node and value is its parent
        self.parent = [-1] * n
        # index is dfs order and value is original node
        self.order_to_node = [-1] * n
        # index is original node and value is its depth
        self.node_depth = [0] * n
        # index is dfs order and value is its depth
        self.order_depth = [0] * n
        # the order of original node visited in the total backtracking path
        self.euler_order = []
        # the pos of original node first appears in the euler order
        self.euler_in = [-1] * n
        # the pos of original node last appears in the euler order
        self.euler_out = [-1] * n  # 每个原始节点再欧拉序中最后出现的位置
        self.build(dct, root)
        return

    def build(self, dct, root):
        """build dfs order and euler order and relative math.info"""
        order = 0
        stack = [(root, -1)]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                self.euler_order.append(i)
                self.start[i] = order
                self.order_to_node[order] = i
                self.end[i] = order
                self.order_depth[order] = self.node_depth[i]
                order += 1
                stack.append((~i, fa))
                for j in dct[i]:
                    if j != fa:
                        # the order of son nodes can be assigned for lexicographical order
                        self.parent[j] = i
                        self.node_depth[j] = self.node_depth[i] + 1
                        stack.append((j, i))
            else:
                i = ~i
                if i != root:
                    self.euler_order.append(self.parent[i])
                if self.parent[i] != -1:
                    self.end[self.parent[i]] = self.end[i]
        for i, num in enumerate(self.euler_order):
            # pos of euler order for every original node
            self.euler_out[num] = i
            if self.euler_in[num] == -1:
                self.euler_in[num] = i
        return
