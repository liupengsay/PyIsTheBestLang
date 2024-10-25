import math
import random
import unittest

from src.graph.union_find.template import UnionFind
from src.tree.tree_diameter.template import TreeDiameter


class TestGeneral(unittest.TestCase):

    def test_tree_dp(self):
        random.seed(2024)
        for _ in range(100):
            n = 100
            edges = []
            for i in range(1, n):
                j = random.randint(0, i - 1)
                edges.append((i, j, random.randint(1, 10 ** 9)))
            uf = UnionFind(n)
            for x, y, _ in edges:
                uf.union(x, y)
            assert uf.part == 1 and len(edges) == n - 1

            ans1 = math.inf
            for i in range(n - 1):
                uf = UnionFind(n)
                for j in range(n - 1):
                    if i != j:
                        x, y, _ = edges[j]
                        uf.union(x, y)
                group = uf.get_root_part()
                assert uf.part == 2
                cur = []
                for g in group:
                    lst = group[g]
                    m = len(lst)
                    ind = {num: i for i, num in enumerate(lst)}
                    dct = [[] for _ in range(m)]
                    for index, (x, y, w) in enumerate(edges):
                        if index == i:
                            continue
                        if uf.find(x) == g:
                            dct[ind[x]].append((ind[y], w))
                            dct[ind[y]].append((ind[x], w))
                    dia = TreeDiameter(dct)
                    cur.append(dia.get_diameter_info()[-1])
                ans1 = min(ans1, max(cur) - min(cur))

            dct = [[] for _ in range(n)]
            for i, j, w in edges:
                dct[i].append((j, w))
                dct[j].append((i, w))
            father = [-1] * n
            stack = [(0, -1)]
            sub = [[0, 0, 0] for _ in range(n)]
            dia = [0] * n
            while stack:
                x, fa = stack.pop()
                if x >= 0:
                    stack.append((~x, fa))
                    for y, w in dct[x]:
                        if y != fa:
                            stack.append((y, x))
                            father[y] = x
                else:
                    x = ~x
                    a = b = c = d = 0
                    for y, ww in dct[x]:
                        if y != fa:
                            for w in sub[y][:1]:
                                if w + ww >= a:
                                    a, b, c = w + ww, a, b
                                elif w + ww >= b:
                                    b, c = w + ww, b
                                elif w + ww >= c:
                                    c = w + ww
                            d = max(d, dia[y])
                    sub[x] = [a, b, c]
                    d = max(d, a + b)
                    dia[x] = d

            ans2 = math.inf
            stack = [(0, -1, 0, 0)]
            while stack:
                x, fa, pre, pre_dia = stack.pop()
                a, b, c = sub[x]
                # print("x, a, b, c, pre, pre_dia", x, a, b, c, pre, pre_dia)
                aa = bb = -math.inf
                for y, _ in dct[x]:
                    if y != fa:
                        dd = dia[y]
                        if dd >= aa:
                            aa, bb = dd, aa
                        elif dd >= bb:
                            bb = dd
                for y, w in dct[x]:
                    if y != fa:
                        down = dia[y]
                        if sub[y][0] == a - w:
                            up = max(pre + b, b + c, pre_dia)
                            nex = max(pre, b) + w
                            nex_dia = max(pre_dia, pre + b, b + w, b + c)
                        elif sub[y][0] == b - w:
                            up = max(pre + a, a + c, pre_dia)
                            nex = max(pre, a) + w
                            nex_dia = max(pre_dia, pre + a, a + w, a + c)
                        else:
                            up = max(pre + a, a + b, pre_dia)
                            nex = max(pre, a) + w
                            nex_dia = max(pre_dia, pre + a, a + w, a + b)
                        if dia[y] == aa:
                            up = max(up, bb)
                            nex_dia = max(nex_dia, bb)
                        else:
                            up = max(up, aa)
                            nex_dia = max(nex_dia, aa)
                        ans2 = min(ans2, abs(up - down))
                        # print("y, up, down", y, up, down)
                        stack.append((y, x, nex, nex_dia))

            # print(edges, ans1, ans2)
            assert ans1 == ans2
        return


if __name__ == '__main__':
    unittest.main()
