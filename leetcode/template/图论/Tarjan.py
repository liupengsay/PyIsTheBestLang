from typing import List


def check_graph(edge: List[list], n):
    """

    :param edge: 边连接关系
    :param n: 节点数
    :return:
    """
    # 访问序号与根节点序号
    visit = [0] * n
    root = [0] * n
    # 割点
    cut_node = []
    # 割边
    cut_edge = []
    # 强连通分量子树
    sub_group = []

    # 中间变量
    stack = []
    index = 1

    def tarjan(i, father):
        nonlocal index
        visit[i] = root[i] = index
        index += 1
        stack.append(i)
        child = 0
        for j in edge[i]:
            if j != father:
                if not visit[j]:
                    child += 1
                    tarjan(j, i)
                    root[i] = min(root[i], root[j])
                    # 割边 low[i] < dfn[i]
                    if visit[i] < root[j]:
                        cut_edge.append(sorted([i, j]))
                    # 两种情况下才为割点 low[i] <= dfn[i]
                    if father != -1 and visit[i] <= root[j]:
                        cut_node.append(i)
                    elif father == -1 and child >= 2:
                        cut_node.append(i)
                elif j in stack:
                    root[i] = min(root[i], visit[j])

        if root[i] == visit[i]:
            lst = []
            while stack[-1] != i:
                lst.append(stack.pop())
            lst.append(stack.pop())
            r = min(root[ls] for ls in lst)
            for ls in lst:
                root[ls] = r
            lst.sort()
            sub_group.append(lst)
        return

    for k in range(n):
        if not visit[k]:
            tarjan(k, -1)
    cut_edge.sort()
    cut_node.sort()
    sub_group.sort()
    return cut_edge, cut_node, sub_group


def test_undirected_graph():
    # 无向无环图
    edge = [[1, 2], [0, 3], [0, 3], [1, 2]]
    n = 4
    cut_edge, cut_node, sub_group = check_graph(edge, n)
    assert not cut_edge
    assert not cut_node
    assert sub_group == [[0, 1, 2, 3]]

    # 无向有环图
    edge = [[1, 2, 3], [0, 2], [0, 1], [0]]
    n = 4
    cut_edge, cut_node, sub_group = check_graph(edge, n)
    assert cut_edge == [[0, 3]]
    assert cut_node == [0]
    assert sub_group == [[0, 1, 2], [3]]

    # 无向有环图
    edge = [[1, 2], [0, 2], [0, 1, 3], [2]]
    n = 4
    cut_edge, cut_node, sub_group = check_graph(edge, n)
    assert cut_edge == [[2, 3]]
    assert cut_node == [2]
    assert sub_group == [[0, 1, 2], [3]]

    # 无向有自环图
    edge = [[1, 2], [0, 2], [0, 1, 3], [2, 3]]
    n = 4
    cut_edge, cut_node, sub_group = check_graph(edge, n)
    assert cut_edge == [[2, 3]]
    assert cut_node == [2]
    assert sub_group == [[0, 1, 2], [3]]
    return


def test_directed_graph():
    # 有向无环图
    edge = [[1, 2], [], [3], []]
    n = 4
    cut_edge, cut_node, sub_group = check_graph(edge, n)
    assert cut_edge == [[0, 1], [0, 2], [2, 3]]
    assert cut_node == [0, 2]
    assert sub_group == [[0], [1], [2], [3]]

    edge = [[1, 2], [2], [3], []]
    n = 4
    cut_edge, cut_node, sub_group = check_graph(edge, n)
    assert cut_edge == [[0, 1], [1, 2], [2, 3]]
    assert cut_node == [1, 2]
    assert sub_group == [[0], [1], [2], [3]]

    # 有向有环图
    edge = [[1, 2], [2], [0, 3], []]
    n = 4
    cut_edge, cut_node, sub_group = check_graph(edge, n)
    assert cut_edge == [[2, 3]]
    assert cut_node == [2]
    assert sub_group == [[0, 1, 2], [3]]
    return


if __name__ == '__main__':
    test_directed_graph()
    test_undirected_graph()
