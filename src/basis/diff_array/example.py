import random
import unittest

from src.basis.diff_array.template import DiffArray, DiffMatrix, PreFixSumMatrix


class TestGeneral(unittest.TestCase):

    def test_diff_array_range(self):
        dar = DiffArray()
        n = 3
        shifts = [[0, 1, 1], [1, 2, -1]]
        diff = dar.get_diff_array(n, shifts)
        assert diff == [1, 0, -1]

        n = 3
        shifts = [1, 2, 3]
        pre = dar.get_array_prefix_sum(n, shifts)
        assert pre == [0, 1, 3, 6]

        left = 1
        right = 2
        assert dar.get_array_range_sum(pre, left, right) == 5
        return

    def test_diff_array_matrix(self):
        dam = DiffMatrix()
        m = 3
        n = 3
        shifts = [[1, 2, 1, 2, 1], [2, 3, 2, 3, 1],
                  [2, 2, 2, 2, 2], [1, 1, 3, 3, 3]]
        diff = [[1, 1, 3], [1, 4, 1], [0, 1, 1]]
        assert dam.get_diff_matrix(m, n, shifts) == diff

        shifts = [[1, 2, 1, 2, 1], [2, 3, 2, 3, 1],
                  [2, 2, 2, 2, 2], [1, 1, 3, 3, 3]]
        shifts = [[x - 1 for x in ls[:-1]] + [ls[-1]] for ls in shifts]
        assert dam.get_diff_matrix2(m, n, shifts) == diff

        random.seed(2023)
        for _ in range(10):
            m = n = 2000
            nums = [[0] * n for _ in range(m)]
            shifts = []
            for _ in range(100):
                x1 = 0
                y1 = 0
                x2 = m - 1
                y2 = n - 1
                num = random.randint(0, n)
                for i in range(x1, x2 + 1):
                    for j in range(y1, y2 + 1):
                        nums[i][j] += num
                shifts.append([x1, x2, y1, y2, num])
            assert nums == dam.get_diff_matrix3(m, n, shifts)
        return

    def test_pre_fix_sum_matrix(self):
        diff = [[1, 1, 3], [1, 4, 1], [0, 1, 1]]
        pre = PreFixSumMatrix(diff)
        assert pre.pre == [[0, 0, 0, 0], [0, 1, 2, 5], [0, 2, 7, 11], [0, 2, 8, 13]]

        xa, ya, xb, yb = 1, 1, 2, 2
        assert pre.query(xa, ya, xb, yb) == sum(sum(d[ya: yb + 1]) for d in diff[xa: xb + 1])
        return


if __name__ == '__main__':
    unittest.main()
