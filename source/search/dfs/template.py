class DFS:
    def __init__(self):
        return

    @staticmethod
    def gen_bfs_order_iteration(dct, root=0):
        # 模板：生成深搜序即 dfs 序以及对应子树编号区间
        n = len(dct)
        for i in range(n):
            dct[i].sort(reverse=True)  # 按照子节点编号从小到大进行遍历
        order = 0
        start = [-1] * n  # 每个原始节点的dfs序号开始点也是node_to_order
        end = [-1] * n  # 每个原始节点的dfs序号结束点
        parent = [-1] * n  # 每个原始节点的父节点
        stack = [[root, -1]]
        depth = [0] * n  # 每个原始节点的深度
        order_to_node = [-1] * n  # 每个dfs序号对应的原始节点编号
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                start[i] = order
                order_to_node[order] = i
                end[i] = order
                order += 1
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:  # 注意访问顺序可以进行调整，比如字典序正序逆序
                        parent[j] = i
                        depth[j] = depth[i] + 1
                        stack.append([j, i])
            else:
                i = ~i
                if parent[i] != -1:
                    end[parent[i]] = end[i]

        return start, end


class DfsEulerOrder:
    def __init__(self, dct, root=0):
        """dfs and euler order of rooted tree which can be used for online point update and query subtree sum"""
        # 模板：dfs序与欧拉序，支持在线区间修改树上边，并且实时查询任意两点树上距离
        n = len(dct)
        for i in range(n):
            dct[i].sort(reverse=True)  # 按照子节点编号从小到大进行遍历
        self.start = [-1] * n  # 每个原始节点的dfs序号开始点也是node_to_order
        self.end = [-1] * n  # 每个原始节点的dfs序号结束点
        self.parent = [-1] * n  # 每个原始节点的父节点
        self.order_to_node = [-1] * n  # 每个dfs序号对应的原始节点编号
        self.node_depth = [0] * n  # 每个原始节点的深度
        self.order_depth = [0] * n  # 每个dfs序号的深度
        self.euler_order = []  # 每个dfs序回溯得到的欧拉序号的原始节点编号
        self.euler_in = [-1] * n  # 每个原始节点再欧拉序中首次出现的位置
        self.euler_out = [-1] * n  # 每个原始节点再欧拉序中最后出现的位置
        self.build(dct, root)
        return

    def build(self, dct, root):
        # 生成dfs序与欧拉序相关信息
        order = 0
        stack = [[root, -1]]
        while stack:
            i, fa = stack.pop()
            if i >= 0:
                self.euler_order.append(i)
                self.start[i] = order
                self.order_to_node[order] = i
                self.end[i] = order
                self.order_depth[order] = self.node_depth[i]
                order += 1
                stack.append([~i, fa])
                for j in dct[i]:
                    if j != fa:  # 注意访问顺序可以进行调整，比如字典序正序逆序
                        self.parent[j] = i
                        self.node_depth[j] = self.node_depth[i] + 1

                        stack.append([j, i])
            else:
                i = ~i
                if i != root:
                    self.euler_order.append(self.parent[i])
                if self.parent[i] != -1:
                    self.end[self.parent[i]] = self.end[i]
        for i, num in enumerate(self.euler_order):
            self.euler_out[num] = i  # 计算欧拉序的位置
            if self.euler_in[num] == -1:
                self.euler_in[num] = i
        return
