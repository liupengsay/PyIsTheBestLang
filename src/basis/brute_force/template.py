

class ViolentEnumeration:
    def __init__(self):
        return

    @staticmethod
    def matrix_rotate(matrix):

        # Rotate matrix 90 degrees clockwise
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                a, b, c, d = matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1], matrix[i][j]
                matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] = a, b, c, d

        # Rotate matrix 90 degrees counterclockwise
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                a, b, c, d = matrix[j][n - i - 1], matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1]
                matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] = a, b, c, d

        return matrix
