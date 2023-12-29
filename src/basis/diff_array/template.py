
class PreFixSumMatrix:
    def __init__(self, mat):
        self.mat = mat
        self.m, self.n = len(mat), len(mat[0])
        self.pre = [[0] * (self.n + 1) for _ in range(self.m + 1)]
        for i in range(self.m):
            for j in range(self.n):
                self.pre[i + 1][j + 1] = self.pre[i][j + 1] + self.pre[i + 1][j] - self.pre[i][j] + mat[i][j]
        return

    def query(self, xa: int, ya: int, xb: int, yb: int) -> int:
        """left up corner is (xa, ya) and right down corner is (xb, yb)"""
        assert 0 <= xa <= xb <= self.m - 1
        assert 0 <= ya <= yb <= self.n - 1
        return self.pre[xb + 1][yb + 1] - self.pre[xb + 1][ya] - self.pre[xa][yb + 1] + self.pre[xa][ya]


class DiffArray:
    def __init__(self):
        return

    @staticmethod
    def get_diff_array(n: int, shifts):
        diff = [0] * n
        for i, j, d in shifts:
            if j + 1 < n:
                diff[j + 1] -= d
            diff[i] += d
        for i in range(1, n):
            diff[i] += diff[i - 1]
        return diff

    @staticmethod
    def get_array_prefix_sum(n: int, lst):
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + lst[i]
        return pre

    @staticmethod
    def get_array_range_sum(pre, left: int, right: int) -> int:
        return pre[right + 1] - pre[left]


class DiffMatrix:
    def __init__(self):
        return

    @staticmethod
    def get_diff_matrix(m: int, n: int, shifts):
        """two dimensional differential array"""
        diff = [[0] * (n + 2) for _ in range(m + 2)]
        # left up corner is (xa, ya) and right down corner is (xb, yb)
        for xa, xb, ya, yb, d in shifts:
            assert 1 <= xa <= xb <= m
            assert 1 <= ya <= yb <= n
            diff[xa][ya] += d
            diff[xa][yb + 1] -= d
            diff[xb + 1][ya] -= d
            diff[xb + 1][yb + 1] += d

        for i in range(1, m + 2):
            for j in range(1, n + 2):
                diff[i][j] += diff[i - 1][j] + diff[i][j - 1] - diff[i - 1][j - 1]

        for i in range(1, m + 1):
            diff[i] = diff[i][1:n + 1]
        return diff[1: m + 1]

    @staticmethod
    def get_diff_matrix2(m, n, shifts):
        """left up corner is (xa, ya) and right down corner is (xb, yb)"""
        diff = [[0] * (n + 1) for _ in range(m + 1)]
        for xa, xb, ya, yb, d in shifts:
            assert 0 <= xa <= xb <= m - 1
            assert 0 <= ya <= yb <= n - 1
            diff[xa][ya] += d
            diff[xa][yb + 1] -= d
            diff[xb + 1][ya] -= d
            diff[xb + 1][yb + 1] += d

        res = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                res[i + 1][j + 1] = res[i + 1][j] + res[i][j + 1] - res[i][j] + diff[i][j]
        return [item[1:] for item in res[1:]]

    @staticmethod
    def get_diff_matrix3(m, n, shifts):
        """left up corner is (xa, ya) and right down corner is (xb, yb)"""
        diff = [0] * ((m + 1) * (n + 1))
        for xa, xb, ya, yb, d in shifts:
            assert 0 <= xa <= xb <= m - 1
            assert 0 <= ya <= yb <= n - 1
            diff[xa * (n + 1) + ya] += d
            diff[xa * (n + 1) + yb + 1] -= d
            diff[(xb + 1) * (n + 1) + ya] -= d
            diff[(xb + 1) * (n + 1) + yb + 1] += d

        res = [0] * ((m + 1) * (n + 1))
        for i in range(m):
            for j in range(n):
                res[(i + 1) * (n + 1) + j + 1] = res[(i + 1) * (n + 1) + j] + res[i * (n + 1) + j + 1] - res[i * (n + 1) + j] + diff[i * (n + 1) + j]
        return [res[i * (n + 1) + 1:(i + 1) * (n + 1)] for i in range(1, m+1)]
