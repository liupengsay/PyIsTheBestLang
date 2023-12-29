

from src.graph.union_find.template import UnionFindLeftRoot


class BinarySearchTree:

    def __init__(self):
        return

    @staticmethod
    def build_with_unionfind(nums):
        """build binary search tree by the order of nums with unionfind"""

        n = len(nums)
        ind = list(range(n))
        ind.sort(key=lambda it: nums[it])
        rank = {idx: i for i, idx in enumerate(ind)}

        dct = [[] for _ in range(n)]
        uf = UnionFindLeftRoot(n)
        post = {}
        for i in range(n - 1, -1, -1):
            x = rank[i]
            if x + 1 in post:
                r = uf.find(post[x + 1])
                dct[i].append(r)
                uf.union(i, r)
            if x - 1 in post:
                r = uf.find(post[x - 1])
                dct[i].append(r)
                uf.union(i, r)
            post[x] = i
        return dct

    @staticmethod
    def build_with_stack(nums):
        """build binary search tree by the order of nums with stack"""

        n = len(nums)

        lst = sorted(nums)
        dct = {num: i + 1 for i, num in enumerate(lst)}
        ind = {num: i for i, num in enumerate(nums)}

        order = [dct[i] for i in nums]
        father, occur, stack = [0] * (n + 1), [0] * (n + 1), []
        deep = [0] * (n + 1)
        for i, x in enumerate(order, 1):
            occur[x] = i

        for x, i in enumerate(occur):
            while stack and occur[stack[-1]] > i:
                if occur[father[stack[-1]]] < i:
                    father[stack[-1]] = x
                stack.pop()
            if stack:
                father[x] = stack[-1]
            stack.append(x)

        for x in order:
            deep[x] = 1 + deep[father[x]]

        dct = [[] for _ in range(n)]
        for i in range(1, n + 1):
            if father[i]:
                u, v = father[i] - 1, i - 1
                x, y = ind[lst[u]], ind[lst[v]]
                dct[x].append(y)
        return dct
