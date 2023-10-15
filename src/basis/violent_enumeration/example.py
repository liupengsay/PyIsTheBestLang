class TestGeneral(unittest.TestCase):

    def test_violent_enumeration(self):
        ve = ViolentEnumeration()
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert ve.matrix_rotate(matrix) == matrix
        return


if __name__ == '__main__':
    unittest.main()
