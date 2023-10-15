class TestGeneral(unittest.TestCase):

    def test_spiral_matrix(self):
        sm = SpiralMatrix()
        nums = [[1, 2, 3, 4], [12, 13, 14, 5], [11, 16, 15, 6], [10, 9, 8, 7]]
        m = len(nums)
        n = len(nums[0])
        for i in range(m):
            for j in range(n):
                assert sm.get_spiral_matrix_num1(
                    m, n, i + 1, j + 1) == nums[i][j]
                assert sm.get_spiral_matrix_num2(
                    m, n, i + 1, j + 1) == nums[i][j]

        nums = [[1, 2, 3, 4, 5, 6], [14, 15, 16, 17, 18, 7],
                [13, 12, 11, 10, 9, 8]]
        m = len(nums)
        n = len(nums[0])
        for i in range(m):
            for j in range(n):
                assert sm.get_spiral_matrix_num1(
                    m, n, i + 1, j + 1) == nums[i][j]
                assert sm.get_spiral_matrix_num2(
                    m, n, i + 1, j + 1) == nums[i][j]

        for _ in range(10):
            m = random.randint(5, 100)
            n = random.randint(5, 100)
            for i in range(m):
                for j in range(n):
                    num = sm.get_spiral_matrix_num1(m, n, i + 1, j + 1)
                    assert sm.get_spiral_matrix_num2(m, n, i + 1, j + 1) == num
                    assert sm.get_spiral_matrix_loc(
                        m, n, num) == [i + 1, j + 1]

        return


if __name__ == '__main__':
    unittest.main()
