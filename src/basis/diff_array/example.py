



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
        # 索引从1开始
        shifts = [[1, 2, 1, 2, 1], [2, 3, 2, 3, 1],
                  [2, 2, 2, 2, 2], [1, 1, 3, 3, 3]]
        diff = [[1, 1, 3], [1, 4, 1], [0, 1, 1]]
        assert dam.get_diff_matrix(m, n, shifts) == diff

        shifts = [[1, 2, 1, 2, 1], [2, 3, 2, 3, 1],
                  [2, 2, 2, 2, 2], [1, 1, 3, 3, 3]]
        shifts = [[x - 1 for x in ls[:-1]] + [ls[-1]] for ls in shifts]
        assert dam.get_diff_matrix2(m, n, shifts) == diff

        pre = dam.get_matrix_prefix_sum(diff)
        assert pre == [[0, 0, 0, 0], [0, 1, 2, 5],
                       [0, 2, 7, 11], [0, 2, 8, 13]]

        xa, ya, xb, yb = 1, 1, 2, 2
        assert dam.get_matrix_range_sum(pre, xa, ya, xb, yb) == 7
        return


if __name__ == '__main__':
    unittest.main()
