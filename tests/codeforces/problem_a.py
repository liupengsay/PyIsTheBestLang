import random
import sys


class FastIO:
    def __init__(self):
        self.random_seed = random.randint(0, 10 ** 9 + 7)
        return

    @staticmethod
    def read_int():
        return int(sys.stdin.readline().strip())

    @staticmethod
    def read_float():
        return float(sys.stdin.readline().strip())

    @staticmethod
    def read_list_ints():
        return list(map(int, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_floats():
        return list(map(float, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_list_ints_minus_one():
        return list(map(lambda x: int(x) - 1, sys.stdin.readline().strip().split()))

    @staticmethod
    def read_str():
        return sys.stdin.readline().strip()

    @staticmethod
    def read_list_strs():
        return sys.stdin.readline().strip().split()

    @staticmethod
    def read_list_str():
        return list(sys.stdin.readline().strip())

    def read_graph(self, n, directed=False):
        dct = [[] for _ in range(n)]
        for _ in range(n - 1):
            i, j = self.read_list_ints_minus_one()
            dct[i].append(j)
            if not directed:
                dct[j].append(i)
        return dct

    @staticmethod
    def st(x):
        return print(x)

    @staticmethod
    def lst(x):
        return print(*x)


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        ans = 0
        n = ac.read_int()
        nums = ac.read_list_ints()

        def range_merge_to_disjoint(left, right):
            nonlocal ans
            if left >= right:
                return

            mid = (left + right) // 2
            range_merge_to_disjoint(left, mid)
            range_merge_to_disjoint(mid + 1, right)

            i, j = left, mid + 1
            k = left
            while i <= mid and j <= right:
                if nums[i] <= nums[j]:
                    arr[k] = nums[i]
                    i += 1
                else:
                    arr[k] = nums[j]
                    j += 1
                    ans += mid - i + 1
                k += 1
            while i <= mid:
                arr[k] = nums[i]
                i += 1
                k += 1
            while j <= right:
                arr[k] = nums[j]
                j += 1
                k += 1

            for i in range(left, right + 1):
                nums[i] = arr[i]
            return

        ans = 0
        arr = [0] * n
        range_merge_to_disjoint(0, n - 1)
        ac.st(ans)
        return


Solution().main()
