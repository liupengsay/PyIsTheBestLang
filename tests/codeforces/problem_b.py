from sys import stdin

inf = 1 << 32


class FastIO:
    def __init__(self):
        self.random_seed = 0
        return

    @staticmethod
    def read_int():
        return int(stdin.readline().rstrip())

    @staticmethod
    def read_float():
        return float(stdin.readline().rstrip())

    @staticmethod
    def read_list_ints():
        return list(map(int, stdin.readline().rstrip().split()))

    @staticmethod
    def read_list_ints_minus_one():
        return list(map(lambda x: int(x) - 1, stdin.readline().rstrip().split()))

    @staticmethod
    def read_str():
        return stdin.readline().rstrip()

    @staticmethod
    def read_list_strs():
        return stdin.readline().rstrip().split()

    def get_random_seed(self):
        import random
        self.random_seed = random.randint(0, 10 ** 9 + 7)
        return

    @staticmethod
    def st(x):
        return print(x)

    @staticmethod
    def lst(x):
        return print(*x)

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def ceil(a, b):
        return a // b + int(a % b != 0)

    @staticmethod
    def accumulate(nums):
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        return pre





class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc287/tasks/abc287_g
        tag: segment_tree|range_sum|dynamic|offline|tree_array|bisect_right
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        q = ac.read_int()
        queries = [ac.read_list_ints() for _ in range(q)]
        nodes = set()
        for a, _ in nums:
            nodes.add(a)
        for lst in queries:
            if lst[0] == 1:
                nodes.add(lst[2])
        nodes = sorted(nodes)
        ind = {num: i + 1 for i, num in enumerate(nodes)}
        n = len(nodes)
        tree1 = PointAddRangeSum(n)
        tree2 = PointAddRangeSum(n)
        tot1 = tot2 = 0
        for a, b in nums:
            tree1.point_add(ind[a], b)
            tree2.point_add(ind[a], a * b)
            tot1 += b
            tot2 += a * b

        for lst in queries:
            if lst[0] < 3:
                x, y = lst[1:]
                x -= 1
                a, b = nums[x]
                tree1.point_add(ind[a], -b)
                tree2.point_add(ind[a], -a * b)
                tot1 -= b
                tot2 -= a * b
                if lst[0] == 1:
                    nums[x][0] = y
                else:
                    nums[x][1] = y
                a, b = nums[x]
                tree1.point_add(ind[a], b)
                tree2.point_add(ind[a], a * b)
                tot1 += b
                tot2 += a * b
            else:
                x = lst[1]
                if tot1 < x:
                    ac.st(-1)
                    continue
                i = tree1.bisect_right(tot1 - x)
                ans = tree2.range_sum(1, i) if i else 0
                rest = tot1 - x - tree1.range_sum(1, i) if i else tot1 - x
                ans += rest * nodes[i]
                ac.st(tot2 - ans)
        return


Solution().main()
